import pickle
from MODEL import create_model

X = pickle.load(open("data/x.db","rb"))
Y = pickle.load(open("data/y.db","rb"))
IMG_SIZE = 100
CATEGORIES = ["happy","not happy"]
X = X/255.0




model = create_model()

model.fit(X,Y,batch_size = 32,epochs = 5,validation_split = 0.1)

model.save_weights("data/model_weights")
