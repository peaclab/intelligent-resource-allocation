import sys
sys.path.append('/projectnb/peaclab-mon/boztop/intelligent-resource-allocation/src')
from preprocessing.preprocessor_factory import PreprocessorFactory

def try_preprocessing():
    available_datasets = ["Fugaku", "Eagle", "Sandia", "BU", "M100"]
    
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
            
            # Get the appropriate preprocessor using the factory
            preprocessor = PreprocessorFactory.get_preprocessor(user_input)
            
            data = preprocessor.preprocess_data()

            print(f"Successfully processed {user_input} dataset\n")

            continue_choice = input("Process another dataset? (y/n): ").strip()
            if continue_choice.lower() != 'y':
                print("Exiting program.")
                break
                
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error processing {user_input}: {e}")

if __name__ == "__main__":
    try_preprocessing()