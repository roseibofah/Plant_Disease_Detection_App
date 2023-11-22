#Library imports
import numpy as np
import streamlit as st
import cv2 as cv2
from keras.models import load_model

#Loading the Model
model = load_model('D:/ML App/Plant Detection App/Plant_Disease/plant_disease.h5')

#Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#Setting Title of App
st.title("APP For Plant Disease Detection")
st.header("--Buit with a CNN Algorithm--")
st.subheader("......For 3 plants: corn, potato and tomatoe......")

st.markdown("Upload an image of the plant leaf")
st.sidebar.info('Keep uploading new images!')
st.image("./tomatoe_flower.jpg", caption='E.g. Tomatoe flower')


#Uploading the dog image
plant_image = st.file_uploader("Choose your OWN image...", type="jpg")
st.write("Now time to HIT the predict BELOW to determine the disease type ---")
submit = st.button('Predict')
#On predict button click
if submit:


    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
