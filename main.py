# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Leonard Umoru',
#     license='',
# )

import logging
import traceback
from src.data.make_dataset import load_data
from src.visualization.visualize import Cluster, Silhouette_plot
from src.models.train_and_predict_model import train_predict_Kmodel

# Set up logging
logging.basicConfig(
    filename="mall_clustering_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

if __name__ == "__main__":
    try:
        logging.info("Mall Clustering Pipeline started.")

        # Load and preprocess the data
        data_path = "data/raw/mall_customers.csv"
        try:
            df = load_data(data_path)
            logging.info(f"Data loaded successfully from {data_path}")
        except Exception as e:
            logging.error("Error loading dataset.")
            logging.error(traceback.format_exc())
            raise

        # Train clustering model
        try:
            kmodel = train_predict_Kmodel(df)
            logging.info("KMeans model trained successfully.")
        except Exception as e:
            logging.error("Error training KMeans model.")
            logging.error(traceback.format_exc())
            raise

        # Generate silhouette plot
        try:
            Silhouette_plot(kmodel)
            logging.info("Silhouette plot generated.")
        except Exception as e:
            logging.warning("Failed to generate silhouette plot.")
            logging.warning(traceback.format_exc())

        # Output model summary
        try:
            print(kmodel)
            logging.info("Model printed to console.")
        except Exception as e:
            logging.warning("Failed to print the model.")
            logging.warning(traceback.format_exc())

        logging.info("Mall Clustering Pipeline completed successfully.")

    except Exception as e:
        logging.critical("Pipeline terminated due to an unrecoverable error.")
        logging.critical(traceback.format_exc())

