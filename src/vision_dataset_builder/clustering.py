from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class EmbeddingClusterer:
    """
    Cluster image embeddings using MiniBatch KMeans.

    For each cluster, the image closest to the cluster
    center can be selected as a representative sample.
    """

    def __init__(
        self,
        embedding_file,
        filename_file,
        output_dir,
        n_clusters=100,
        batch_size=1024,
        random_state=42,
    ):
        self.embedding_file = Path(embedding_file)
        self.filename_file = Path(filename_file)
        self.output_dir = Path(output_dir)

        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_data(self):
        """Load embeddings and their corresponding filenames."""

        if not self.embedding_file.exists():
            raise FileNotFoundError(
                f"Embedding file not found: "
                f"{self.embedding_file}"
            )

        if not self.filename_file.exists():
            raise FileNotFoundError(
                f"Filename file not found: "
                f"{self.filename_file}"
            )

        embeddings = np.load(
            self.embedding_file
        )

        with open(
            self.filename_file,
            "r",
            encoding="utf-8"
        ) as file:
            filenames = json.load(file)

        if len(embeddings) != len(filenames):
            raise ValueError(
                "Number of embeddings does not match "
                "number of filenames."
            )

        return embeddings, filenames

    def cluster(self, embeddings):
        """Run MiniBatch KMeans clustering."""

        if len(embeddings) < self.n_clusters:
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) "
                f"must be greater than or equal to "
                f"n_clusters ({self.n_clusters})."
            )

        print("\nRunning MiniBatch KMeans...")

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init=10,
        )

        labels = kmeans.fit_predict(
            embeddings
        )

        centers = kmeans.cluster_centers_

        print("Clustering complete.")

        return labels, centers

    def select_representatives(
        self,
        embeddings,
        filenames,
        labels,
        centers,
    ):
        """
        Select the image closest to each cluster center.
        """

        representatives = []
        cluster_statistics = []

        for cluster_id in range(
            self.n_clusters
        ):

            indices = np.where(
                labels == cluster_id
            )[0]

            if len(indices) == 0:
                continue

            cluster_embeddings = (
                embeddings[indices]
            )

            distances = np.linalg.norm(
                cluster_embeddings
                - centers[cluster_id],
                axis=1,
            )

            best_local_index = np.argmin(
                distances
            )

            best_index = indices[
                best_local_index
            ]

            representatives.append({
                "cluster": cluster_id,
                "representative": filenames[
                    best_index
                ],
                "cluster_size": len(indices),
                "distance_to_center": float(
                    distances[best_local_index]
                ),
            })

            cluster_statistics.append({
                "cluster": cluster_id,
                "size": len(indices),
            })

        return (
            representatives,
            cluster_statistics
        )

    def save_results(
        self,
        labels,
        centers,
        representatives,
        cluster_statistics,
    ):
        """Save clustering results."""

        np.save(
            self.output_dir / "labels.npy",
            labels
        )

        np.save(
            self.output_dir / "cluster_centers.npy",
            centers
        )

        pd.DataFrame(
            representatives
        ).to_csv(
            self.output_dir
            / "representative_mapping.csv",
            index=False,
        )

        pd.DataFrame(
            cluster_statistics
        ).to_csv(
            self.output_dir
            / "cluster_statistics.csv",
            index=False,
        )

    def run(self):
        """Run the complete clustering pipeline."""

        print("=" * 60)
        print("EMBEDDING CLUSTERING")
        print("=" * 60)

        embeddings, filenames = (
            self.load_data()
        )

        print(
            f"Images: {len(filenames)}"
        )

        print(
            f"Embedding shape: "
            f"{embeddings.shape}"
        )

        labels, centers = self.cluster(
            embeddings
        )

        (
            representatives,
            cluster_statistics,
        ) = self.select_representatives(
            embeddings,
            filenames,
            labels,
            centers,
        )

        self.save_results(
            labels,
            centers,
            representatives,
            cluster_statistics,
        )

        cluster_sizes = [
            item["size"]
            for item in cluster_statistics
        ]

        print("\n" + "=" * 60)
        print("CLUSTERING COMPLETE")
        print("=" * 60)
        print(
            f"Images: {len(filenames)}"
        )
        print(
            f"Clusters requested: "
            f"{self.n_clusters}"
        )
        print(
            f"Clusters populated: "
            f"{len(cluster_statistics)}"
        )
        print(
            f"Representatives: "
            f"{len(representatives)}"
        )

        if cluster_sizes:
            print(
                f"Largest cluster: "
                f"{max(cluster_sizes)}"
            )
            print(
                f"Smallest cluster: "
                f"{min(cluster_sizes)}"
            )
            print(
                f"Average cluster size: "
                f"{np.mean(cluster_sizes):.2f}"
            )

        print(
            f"\nResults saved to: "
            f"{self.output_dir}"
        )
        print("=" * 60)