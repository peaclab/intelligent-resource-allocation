import sys
sys.path.append('/projectnb/peaclab-mon/boztop/intelligent-resource-allocation/src')
from preprocessing.preprocessor_factory import PreprocessorFactory
from clustering.kmeans_clustering import KMeansClustering
from ml.train_models import ModelTrainer
from ml.test_models import ModelTester

def try_ml():
    available_datasets = ["Fugaku", "Eagle", "Sandia", "BU", "M100"]
    dataset_latest_dates = {
                "Fugaku": "2024-04-30",
                "Eagle": "2023-02-01",
                "Sandia": "2024-09-23",
                "BU": "2023-12-31",
                "M100": "2021-12-31"
            }

    dataset_recommended_number_of_clusters = {
                "Fugaku": 4,
                "Eagle": 18,
                "Sandia": 18,
                "BU": 18,
                "M100": 6}        
    
    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")
    
    while True:
        print("\nEnter dataset name (or 'q' to quit): ", end="")
        user_input = input().strip()
        
        if user_input.lower() == 'q':
            print("Exiting program.")
            break
        
        try:
            print(f"\nProcessing dataset: {user_input}")
            preprocessor = PreprocessorFactory.get_preprocessor(user_input)
            data = preprocessor.preprocess_data()
            print(f"Successfully processed {user_input} dataset\n")

            
            if user_input in dataset_latest_dates:
                end_date = dataset_latest_dates[user_input]
                print(f"The latest job date for {user_input} is {end_date}.")
            else:
                print(f"No latest job date available for dataset {user_input}.")

            days_to_train = input("Enter the number of days you want to train the model on: ").strip()
            print(f"Training the model on {days_to_train} days of data.")
            days_to_train = int(days_to_train)


            num_clusters = input(f"Enter the number of clusters you want to create (recommended: {dataset_recommended_number_of_clusters.get(user_input, 'N/A')}): ").strip()
            
            try:
                num_clusters = int(num_clusters)
                clustering_model = KMeansClustering(num_clusters)
                sub_dfs, cluster_centers = clustering_model.create_sub_dataframes(data, preprocessor.train_features,days_to_train)
                print(f"Successfully created {num_clusters} clusters.")
            except ValueError:
                print("Invalid number of clusters. Please enter an integer value.")

            model_type = input("Enter the model type (xgboost or rf): ").strip()
            resampling = input("Enable resampling (y/n)? ").strip().lower() == 'y'

            if user_input == 'Sandia': 
                target_feature = input("Enter the target feature: (choices are max_cpu_node_usage, mean_cpu_node_usage, max_mem_usage, mean_mem_usage, and execution_time) ").strip() 
            elif user_input == 'Fugaku':
                target_feature = input("Enter the target feature: (choices are duration, cnumut, and mmszu) ").strip() 
            else:
                print('You are going to predict the execution time, good luck!')
                target_feature = preprocessor.target_feature    
                
            train_models = ModelTrainer(model_type, resampling)
            models, biases = train_models.train_model_per_cluster(sub_dfs, preprocessor.train_features, target_feature)
            print("Successfully trained models.")

            bias_type = input("Enter the bias type (none, mean, mad, std_dev, two_sigma): ").strip()
            model_tester = ModelTester(models, biases, cluster_centers, bias_type, model_type)

            while True:
                input_data = {}
                for feature in preprocessor.train_features:
                    input_data[feature] = input(f"Enter value for {feature}: ").strip()

                try:
                    result = model_tester.test_single_input(input_data, preprocessor.train_features, target_feature)
                    print(f"Prediction result for target feature of {target_feature}:")
                    for key, value in result.items():
                        print(f"{key}: {value}")
                except Exception as e:
                    print(f"Unexpected error processing {user_input}: {e}")

            print("Thank you! Exiting program.")
            break

                
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error processing {user_input}: {e}")

if __name__ == "__main__":
    try_ml()