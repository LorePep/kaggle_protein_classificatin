import click
import numpy as np
import os
import pandas as pd
from keras.models import Sequential, load_model, Model
from tqdm import tqdm

from data import data_generator
from model import f1


@click.command(help="Create dataset.")
@click.option("-m", "--model", prompt=True, type=str)
@click.option("-s", "--sample-file-path", prompt=True, type=str)
@click.option("-t", "--test-path", prompt=True, type=str)
def main(
    model,
    sample_file_path,
    test_path,
):
    submit = pd.read_csv(sample_file_path)
    model = load_model(model, custom_objects={"f1": f1})

    input_shape = (256,256,4)


    predicted = []
    for name in tqdm(submit['Id']):
        path = os.path.join(test_path, name)
        image = data_generator.load_image(path, input_shape)
        score_predict = model.predict(image[np.newaxis])[0]
        max_val = np.max(score_predict)
        label_predict = np.arange(28)[score_predict>=0.65*max_val]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    submit.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
