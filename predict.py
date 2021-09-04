#
# TEST.PY
# Created by: AHMET CELIK
#

import json
from get_input_args import get_predict_args
from ai_methods import load_checkpoint, predict_method

def main():
    predict_args = get_predict_args()
    
    predict_flag = True
    json_flag = True
    try:
        model = load_checkpoint(predict_args.checkpoint)
        print("Model is loaded from {}.".format(predict_args.checkpoint))
    except:
        print("Model cannot be loaded from {}!".format(predict_args.checkpoint))
        predict_flag = False
    
    try:
        with open(predict_args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    except:
        print("Category names cannot be loaded from {}!".format(predict_args.category_names))
        json_flag = False
    
    if predict_flag:
        my_device = "cuda" if predict_args.gpu else "cpu"
        try:
            probs, classes = predict_method(predict_args.path_to_image, model, predict_args.top_k, my_device)
        except:
            print("Cannot open {}!".format(predict_args.path_to_image))
            return
        
        print()
        if json_flag:
            print("Predicted Class Names & Probabilities:")
            classes_names = [cat_to_name[str(class_num)] for class_num in classes]
            result_list = zip(classes_names, probs)
        else:
            print("Predicted Class Numbers & Probabilities:")
            result_list = zip(classes, probs)
        
        for el in result_list:
            print("{}: {}".format(el[0], el[1]))
    
    return result_list

# Call to main function to run the program
if __name__ == "__main__":
    main()