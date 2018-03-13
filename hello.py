import keras
import numpy as np
import flask
from flask_cors import CORS, cross_origin
import io

app = flask.Flask(__name__)
CORS(app)
model = None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    fname = 'my_model.h5'
    model = keras.models.load_model(fname)


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
        # initialize the data dictionary that will be returned from the
        # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.is_json:
            # read json
            content = flask.request.get_json()
            
            # print(content['data'])

            arr = np.array([content['data']])

            # classify the input image and then initialize the list
            # of predictions to return to the client
            pred = model.predict(arr)

            data["predictions"] = np.asarray(pred).tolist()

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
