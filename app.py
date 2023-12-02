import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# getting names of the classes
with open('classes.txt', 'r') as f:
    class_names=[i for i in f.readlines()]
class_names=list(map(lambda x:x.strip(), class_names))
dic={}
for i in range(len(class_names)):
    dic[i]=class_names[i]
print(class_names)


# loading the model

PATH='MyResNet.pth'

# Load the pretrained model
model = models.resnet18(pretrained=True)

# Modify the fully connected layer (fc)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 9))  # 9 is the number of output classes

# Load the state dictionary
state_dict = torch.load(PATH, map_location=torch.device('cpu'))

# Update the model parameters
model.load_state_dict(state_dict)
model.eval()



# for the page title and page icon

plant_icon_path=Image.open('app-icon.png')
st.set_page_config(
    page_title="Plant nutrition deficiency ",
    page_icon=plant_icon_path,
    layout="centered",
    initial_sidebar_state="auto"
)


st.title('Hello there!')



# css code for styling 

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



st.text('Worried about your plant health? Get it checked up real quick.')

# upload image

file=st.file_uploader('',type=['jpeg', 'png', 'jpg'])


# display image
if file is not None:
    image = Image.open(file)
    st.image(image, caption='Image uploaded.', use_column_width=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)


# making predictions
if file is not None:
    with torch.no_grad():
        output = model(input_batch)
    _, predicted = torch.max(output, 1)

    # st.write('The Predicted result is: ', ' '.join('%5s' % predicted[0]))
    prediction=dic[int(predicted[0])]
    if prediction=='Healthy':
        st.write('Your banana plant is ',dic[int(predicted[0])])
    else:
        st.write('Your banana plant have ',dic[int(predicted[0])],'deficiency!')
    
    
    

