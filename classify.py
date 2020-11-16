#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
#import tensorflow as tf
#import tensorflow.keras as keras
import tflite_runtime.interpreter as tflite

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=1)
    #y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    #with tf.device('/gpu:0'):
    with open(args.output, 'w') as output_file:
        json_file = open(args.model_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        #model = keras.models.model_from_json(loaded_model_json)
        #model.load_weights(args.model_name+'.h5')
        #model.compile(loss='categorical_crossentropy',
        #              optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        #             metrics=['accuracy'])
        
        #converter = tf.lite.TFLiteConverter.from_keras_model(model)
        #tflite_model = converter.convert()
        
        tf_interpreter = tflite.Interpreter(args.model_name+".tflite")
        tf_interpreter.allocate_tensors()

        input_tf = tf_interpreter.get_input_details()
        output_tf = tf_interpreter.get_output_details()
        
        # Save the model.
        #with open(args.model_name+'.tflite', 'wb') as f:
        #  f.write(tflite_model)
        files = os.listdir(args.captcha_dir)
        files = sorted(files)

        for x in files:
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data, dtype=numpy.float32) / 255.0
            #image = numpy.array(rgb_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])
            #prediction = model.predict(image)
            
            #output_file.write(x + "," + decoded_symbol + "\n")
            
            tf_interpreter.set_tensor(input_tf[0]['index'],image)
            tf_interpreter.invoke()
            prediction = []
            for output_node in output_tf:
                prediction.append(tf_interpreter.get_tensor(output_node['index']))
            prediction = numpy.reshape(prediction,(len(output_tf),-1))
            decoded_symbol = decode(captcha_symbols, prediction)
            decoded_symbol = decoded_symbol.replace('1', '<')
            decoded_symbol = decoded_symbol.replace('2', '>')
            decoded_symbol = decoded_symbol.replace('3', '*')
            decoded_symbol = decoded_symbol.replace('4', '?')
            decoded_symbol = decoded_symbol.replace('5', '/')
            decoded_symbol = decoded_symbol.replace('6', '\\')
            decoded_symbol = decoded_symbol.replace('7', '|')
            decoded_symbol = decoded_symbol.replace('8', ':')
            decoded_symbol = decoded_symbol.replace('9', '\"')
            decoded_symbol = decoded_symbol.replace(' ', '')
            output_file.write(x + "," + decoded_symbol + "\n")
            
            print('Classified ' + x)

        #with open(args.output, 'rb') as output_file:
        #    content = output_file.read()

        #content = content.replace(b'\r\n', b'\n')

        #with open(args.output, 'wb') as output_file:
        #    output_file.write(content)
                
if __name__ == '__main__':
    main()
