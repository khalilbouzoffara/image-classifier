import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import argparse
import numpy as np
import json

image_size = 224

def process_image(image):
    tensor_image = tf.image.convert_image_dtype(image, dtype=tf.int16, saturate=False)
    resized_image = tf.image.resize(image,(image_size,image_size)).numpy()
    norm_image = resized_image/255
    return norm_image

def getClassesNames(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    
    return class_names
    
def predict(image_path, model_path, top_k, all_class_names):
    
    #Load model
    loaded_model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    
    #Process the image
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    
    #Make prediction
    prob_preds = loaded_model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()
    
    #top_k preds
    values, indices= tf.math.top_k(prob_preds, k=top_k)
    probs=values.numpy().tolist()
    classes=indices.numpy().tolist()
    
    #get class_names
    json_file = 'label_map.json'
    class_names = getClassesNames(json_file)
    
    #Label preds
    label_names = [class_names[str(label+1)] for label in classes]
    
    return probs , label_names

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path")
    parser.add_argument("saved_model")
    parser.add_argument("--top_k", required = False)
    parser.add_argument("--category_names",required = False, default = "label_map.json")
    args = parser.parse_args()
    
    class_names = getClassesNames(args.category_names)
    probs , label_names = predict(args.image_path, args.saved_model, args.top_k, class_names)
    print(probs)
    print(label_names)