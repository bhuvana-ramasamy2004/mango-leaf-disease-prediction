import streamlit as st
import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import base64
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image
import matplotlib.image as mpimg
import streamlit as st
import base64

# ---- CHOOSE INPUT IMAGE

uploaded_file = st.file_uploader("Choose a file")

import pickle
with open('file.pickle', 'wb') as f:
    pickle.dump(uploaded_file, f)
            


# aa = st.button("UPLOAD IMAGE")

if uploaded_file is None:
    
    st.text("Please upload an image")

else:
    import numpy as np

    img = mpimg.imread(uploaded_file)
    # st.text(uploaded_file)
    st.image(img,caption="Original Image")
    
    
    #====================== READ A INPUT IMAGE =========================
    
    
    #filename = askopenfilename()
    #img = mpimg.imread(filename)
    #plt.imshow(img)
    #plt.title('Original Image') 
    #plt.axis ('off')
    #plt.show()
    
    #st.write("----------------------------------------------------------------------------------")
    
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300 ,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()

           
    
    #==== GRAYSCALE IMAGE ====
    
    try:            
        gray11 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray11 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray11,cmap="gray")
    plt.axis ('off')
    plt.show()


    #============================ 6. IMAGE SPLITTING ===========================
    
    import os 
    
    from sklearn.model_selection import train_test_split
    
    
    data_1 = os.listdir('F:/CODE/Dataset/Anthracnose')
    
    data_2 = os.listdir('F:/CODE/Dataset/Bacterial Canker/')
    
    data_3 = os.listdir('F:/CODE/Dataset/Cutting Weevil/')
    
    data_4 = os.listdir('F:/CODE/Dataset/Die Back/')
    
    data_5 = os.listdir('F:/CODE/Dataset/Gall Midge/')
    
    data_6 = os.listdir('F:/CODE/Dataset/Healthy/')
    
    data_7 = os.listdir('F:/CODE/Dataset/Powdery Mildew/')
    
    data_8 = os.listdir('F:/CODE/Dataset/Sooty Mould/')
    
    
    
    # ------
    
    
    dot1= []
    labels1 = [] 
    
    
    for img11 in data_1:
            # print(img)
            img_1 = mpimg.imread('Dataset/Anthracnose//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(1)
    
    for img11 in data_2:
            # print(img)
            img_1 = mpimg.imread('Dataset/Bacterial Canker//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(2)
    
    
    for img11 in data_3:
            # print(img)
            img_1 = mpimg.imread('Dataset/Cutting Weevil//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(3)
    
    for img11 in data_4:
            # print(img)
            img_1 = mpimg.imread('Dataset/Die Back//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(4)
    
    
    
    
    for img11 in data_5:
            # print(img)
            img_1 = mpimg.imread('Dataset/Gall Midge//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(5)
    
    for img11 in data_6:
            # print(img)
            img_1 = mpimg.imread('Dataset/Healthy//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(6)
    
    
    for img11 in data_7:
            # print(img)
            img_1 = mpimg.imread('Dataset/Powdery Mildew//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(7)
    
    for img11 in data_8:
            # print(img)
            img_1 = mpimg.imread('Dataset/Sooty Mould//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(8)
    
    
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    print()
    print("-------------------------------------")
    print("       IMAGE SPLITTING               ")
    print("-------------------------------------")
    print()
    
    
    print("Total no of data        :",len(dot1))
    print("Total no of train data  :",len(x_train))
    print("Total no of test data   :",len(x_test))
    
    
 # ----------------- PREDICTION

    Total_length = data_1 + data_2 + data_3 + data_4 +  data_6 + data_7 + data_8
    
    temp_data1  = []
    for ijk in range(0,len(Total_length)):
                # print(ijk)
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray11))
            temp_data1.append(temp_data)
                
    temp_data1 =np.array(temp_data1)
            
    zz = np.where(temp_data1==1)
            
    if labels1[zz[0][0]] == 1:
         
         print("----------------------------------------")
         print("Identified as Disease - ANTHRACNOSE")
         print("----------------------------------------")
         st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as Disease - ANTHRACNOSE"}</h1>', unsafe_allow_html=True)
         
         # Introduction
         st.write("""
        Anthracnose is a common fungal disease that affects a wide variety of plants. It can cause significant damage to crops like beans, tomatoes, peppers, and fruit trees. Below are **10 effective remedies** to help you manage and control anthracnose in your garden or farm.
        """)
        
        # List of Remedies
         remedies = [
            "### 1. Crop Rotation\nRotate crops annually, especially with plants that aren't affected by anthracnose. This reduces the build-up of fungal spores in the soil.",
            "### 2. Pruning and Removing Infected Parts\nRegularly prune infected leaves, stems, or fruit to limit the spread of the disease. Destroy any diseased plant material to prevent spores from overwintering.",
            "### 3. Proper Watering Techniques\nWater plants at the base rather than overhead to avoid wetting the foliage. Early morning watering is best, as it allows the plants to dry out before evening, reducing the chance of fungal growth.",
            "### 4. Improve Air Circulation\nSpace plants properly to allow for adequate airflow. Fungal spores thrive in humid, stagnant environments, so providing good circulation can reduce the disease’s spread.",
            "### 5. Use Copper-based Fungicides\nCopper fungicides (e.g., copper sulfate, copper hydroxide) can be effective in controlling anthracnose. Apply them according to manufacturer instructions, especially during the early stages of disease.",
            "### 6. Apply Sulfur-based Fungicides\nSulfur fungicides can help prevent and control fungal infections, including anthracnose. These are especially useful for fruit trees and other susceptible crops.",
            "### 7. Utilize Biological Controls\nUse beneficial microorganisms such as *Trichoderma* spp. or *Bacillus subtilis*, which can outcompete or inhibit the growth of anthracnose-causing fungi in the soil.",
            "### 8. Use Neem Oil as a Natural Fungicide\nNeem oil can be used to control a variety of fungal diseases, including anthracnose. It also acts as an insect repellent and is safe for beneficial insects.",
            "### 9. Select Resistant Varieties\nWhenever possible, choose plant varieties that are resistant to anthracnose. Some cultivars have been bred for disease resistance and can significantly reduce the impact of the disease.",
            "### 10. Maintain Soil Health\nHealthy soil can improve plant resistance to disease. Use organic matter like compost or mulch to enrich the soil and encourage strong, resilient plants. Ensure proper drainage to prevent waterlogging, which fosters fungal growth."
        ]
        
        # Display remedies
         for remedy in remedies:
            st.markdown(remedy)
        
        # Footer
         st.write("**Note**: Consistent monitoring and integrated management strategies are key to reducing the impact of anthracnose and ensuring plant health.")

         
    elif labels1[zz[0][0]] == 2:
         
         print("----------------------------------------")
         print("Identified as Disease - BACTERIAL CANKER")
         print("----------------------------------------") 
         st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as Disease - BACTERIAL CANKER"}</h1>', unsafe_allow_html=True)

         # Introduction
         st.write("""
        Bacterial Canker is a bacterial disease that affects a variety of plants, especially fruit trees and tomatoes. It can cause lesions on leaves, stems, and fruits, leading to plant wilting, dieback, and sometimes even death. Below are **10 effective remedies** for managing and controlling bacterial canker in your garden or farm.
        """)
        
        # List of Remedies
         remedies = [
            "### 1. Remove Infected Plant Material\nPrune and remove any infected branches, leaves, or fruits. Destroy these parts to prevent the spread of the bacteria to healthy parts of the plant or nearby plants.",
            "### 2. Use Copper-based Fungicides (Bactericides)\nCopper-based products, such as copper sulfate or copper hydroxide, can help manage bacterial infections. Apply these as a preventative treatment during the growing season, especially during the early stages of disease or when environmental conditions are favorable for the spread of the bacteria.",
            "### 3. Practice Good Garden Hygiene\nClean tools, equipment, and hands after handling infected plants. Disinfecting pruning tools with a solution of 10% bleach or rubbing alcohol can prevent the bacteria from spreading to healthy plants.",
            "### 4. Avoid Overhead Irrigation\nAvoid wetting the foliage with overhead irrigation, as bacteria spread more easily in wet conditions. Instead, water at the base of the plant to keep the leaves dry, reducing the chance of bacterial infection.",
            "### 5. Prune During Dry Weather\nPrune plants during dry weather when bacterial activity is low. Wet or rainy conditions can spread the bacteria, so pruning during dry spells reduces the risk of transmission.",
            "### 6. Maintain Proper Spacing and Airflow\nSpace plants properly to ensure good air circulation. Crowded conditions promote high humidity and create an environment conducive to bacterial spread.",
            "### 7. Use Resistant Varieties\nWhen available, plant disease-resistant varieties. Some tomato and fruit tree cultivars have been bred for resistance to bacterial canker and can be an effective way to avoid the disease.",
            "### 8. Disinfect Watering Systems\nIf using drip irrigation or other irrigation systems, clean and disinfect them regularly. Water systems can harbor bacteria, which can then be spread to plants.",
            "### 9. Control Weeds and Debris\nWeeds and plant debris can harbor bacterial pathogens. Keep the area around plants free of weeds and fallen plant matter, as these can act as reservoirs for the bacteria.",
            "### 10. Chemical Treatments (When Necessary)\nIn severe cases, systemic bactericides such as **Streptomycin** or **Oxytetracycline** may be used to treat bacterial canker. These are most effective if applied early, before the disease has spread significantly. Always follow label instructions and consult local agricultural extension services for proper usage."
        ]
        
        # Display remedies
         for remedy in remedies:
            st.markdown(remedy)
        
        # Footer
         st.write("**Note**: Early detection and proper management practices are key to reducing the impact of bacterial canker and protecting your plants.")
            
    elif labels1[zz[0][0]] == 3:
         
         print("----------------------------------------")
         print("Identified as Disease - CUTTING WEEVIL")
         print("----------------------------------------")  
         st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as Disease - CUTTING WEEVIL"}</h1>', unsafe_allow_html=True)

         

        # Introduction
         st.write("""
        Cutting Weevil is a destructive pest that feeds on a variety of plants by cutting through stems, leaves, and roots, often leading to plant death. Common crops affected include cotton, corn, sugarcane, and vegetables. Below are **10 effective remedies** to manage and control cutting weevil infestations in your garden or farm.
        """)
        
        # List of Remedies
         remedies = [
            "### 1. Manual Removal\nInspect plants regularly and manually remove any visible weevils or damaged plant parts. You can also hand-pick adult weevils from the plants early in the morning or late evening when they are most active.",
            "### 2. Use Neem Oil\nNeem oil works as a natural insecticide and can disrupt the weevil's reproductive cycle. It also has anti-feeding properties, which reduces the damage caused by the pests. Apply neem oil to the affected plants, especially on the leaves and stems.",
            "### 3. Insecticidal Soap\nInsecticidal soaps are effective against a variety of pests, including cutting weevils. These soaps suffocate the insects by blocking their breathing pores. Apply them directly to the leaves, stems, and any visible weevil activity.",
            "### 4. Biological Control with Predatory Insects\nIntroduce natural predators of weevils, such as **nematodes** (*Heterorhabditis bacteriophora*). These microscopic worms attack and kill the larvae of cutting weevils in the soil. Beneficial insects like **ladybugs** or **lacewing larvae** can also help control smaller weevil populations.",
            "### 5. Apply Diatomaceous Earth\nDiatomaceous earth is a natural product made from the fossilized remains of diatoms. It works by piercing the exoskeletons of weevils and causing them to dehydrate. Dust the affected areas with diatomaceous earth, particularly around the base of the plants.",
            "### 6. Use Chemical Insecticides\nIn cases of heavy infestations, chemical insecticides such as **pyrethroids** or **carbaryl** can be used to control cutting weevil populations. Always follow the manufacturer's instructions and consider using targeted insecticides that are less harmful to beneficial insects.",
            "### 7. Plant Trap Crops\nPlanting crops that attract cutting weevils away from your main crops can help divert them. **Radishes**, **mustard**, and **clover** can serve as good trap crops, as the weevils will prefer these plants over your main crops.",
            "### 8. Crop Rotation\nPractice crop rotation to reduce the buildup of cutting weevil populations in the soil. Avoid planting the same crops in the same location year after year, as this encourages weevils to stay in the area.",
            "### 9. Tillage and Soil Disruption\nRegular tilling or deep plowing of soil can disrupt the life cycle of the cutting weevil, especially the larvae, which live in the soil. Turning the soil after harvest or in the off-season can help expose and kill weevil larvae.",
            "### 10. Mulching and Covering the Soil\nApplying mulch around the base of plants helps retain moisture and keeps the soil temperature regulated. In addition, **plastic row covers** or **floating row covers** can protect plants from adult weevils that are looking for places to lay eggs."
        ]
        
        # Display remedies
         for remedy in remedies:
            st.markdown(remedy)
        
        # Footer
         st.write("**Note**: Integrated pest management (IPM) that combines cultural, biological, and chemical control methods will be most effective in managing cutting weevil infestations.")
        



    elif labels1[zz[0][0]] == 4:
         
         print("----------------------------------------")
         print("Identified as Disease - DIE BACK")
         print("----------------------------------------") 
         st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as Disease - DIE BACK"}</h1>', unsafe_allow_html=True)

        # Introduction
         st.write("""
        Dieback is a condition that causes plant tissues to die, usually starting from the tips or roots. It can be caused by various factors, including fungal and bacterial infections, pests, or environmental stress. Below are **10 effective remedies** to manage and prevent dieback in your plants.
        """)
        
        # List of Remedies
         remedies = [
            "### 1. Prune Infected Plant Parts\nRegularly inspect plants and prune back any dead, diseased, or damaged branches or stems. Pruning infected tissue helps prevent the spread of pathogens that cause dieback and promotes better airflow around the plant.",
            "### 2. Improve Soil Drainage\nEnsure the soil is well-drained to avoid waterlogged conditions, which can lead to root rot, a common cause of dieback. If necessary, amend the soil with organic matter or sand to improve its structure and drainage.",
            "### 3. Water Properly\nWater plants deeply and infrequently rather than shallow and often. Ensure that the water reaches the root zone, but avoid overhead watering, as it can increase humidity and favor fungal growth. Water at the base of the plant to minimize stress.",
            "### 4. Fungicide Treatment\nIf dieback is caused by a fungal infection, applying a suitable fungicide can help manage the disease. Copper-based fungicides or systemic fungicides containing ingredients like **propiconazole** can be effective at controlling fungal pathogens.",
            "### 5. Insect Control\nInsects such as borers and aphids can weaken plants and contribute to dieback. Use appropriate insecticides or natural predators (like ladybugs for aphids) to control pest populations. Be sure to follow proper application guidelines.",
            "### 6. Promote Healthy Growth\nProvide your plants with proper care to reduce stress and promote resilience. This includes fertilizing them with balanced nutrients, ensuring they get enough sunlight, and avoiding over-pruning, which can weaken them.",
            "### 7. Remove Fallen Debris\nClean up and dispose of fallen plant debris, leaves, and branches. These can harbor pathogens or insects that may contribute to dieback. Keeping the area around the plant clean reduces the likelihood of reinfection.",
            "### 8. Correct Nutrient Deficiencies\nConduct a soil test to determine if your plants have nutrient deficiencies, especially of key elements like nitrogen, phosphorus, or potassium. Deficiencies can weaken the plant, making it more susceptible to dieback. Use appropriate fertilizers to correct deficiencies.",
            "### 9. Improve Air Circulation\nEnsure that plants are spaced properly to allow for good air circulation. Overcrowding can increase humidity around the plant, creating an ideal environment for fungal growth that can lead to dieback.",
            "### 10. Use Protective Coatings for Trunk and Wounds\nAfter pruning or any physical injury, apply a protective wound dressing or tree paint to the cut surfaces. This can help seal off the plant and reduce the entry of pathogens that cause dieback."
        ]
        
        # Display remedies
         for remedy in remedies:
            st.markdown(remedy)
        
        # Footer
         st.write("**Note**: Early intervention and regular monitoring are key to managing dieback disease and ensuring plant health.")
                 
            
    elif labels1[zz[0][0]] == 5:
         
         print("----------------------------------------")
         print("Identified as Disease - GALL MIDGE")
         print("----------------------------------------")      
         st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as Disease - GALL MIDGE"}</h1>', unsafe_allow_html=True)

         



    elif labels1[zz[0][0]] == 6:
          
          print("----------------------------------------")
          print("Identified as  HEALTHY")
          print("----------------------------------------")      
          st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as  HEALTHY"}</h1>', unsafe_allow_html=True)

         
    elif labels1[zz[0][0]] == 7:
         
         print("----------------------------------------")
         print("Identified as Disease - POWDERY MILDEW")
         print("----------------------------------------")   
         st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as Disease - POWDERY MILDEW"}</h1>', unsafe_allow_html=True)

         # Introduction
         st.write("""
        Powdery mildew is a fungal disease that affects many plants, including vegetables, fruits, and ornamental plants. It is characterized by a white or grayish powdery coating on plant surfaces. Below are **10 effective remedies** to help control and prevent powdery mildew in your garden or farm.
        """)
        
        # List of Remedies
         remedies = [
            "### 1. Prune Infected Areas\nRegularly prune and remove any infected leaves, stems, or flowers to prevent the spread of the fungus. Dispose of the infected material properly (do not compost it) to limit the risk of reinfection.",
            "### 2. Improve Air Circulation\nEnsure proper spacing between plants to allow for good airflow. Powdery mildew thrives in high humidity and stagnant air. Pruning to open up the plant structure will help reduce the environmental conditions that favor the fungus.",
            "### 3. Watering Practices\nWater plants at the base, avoiding overhead irrigation, which can spread fungal spores. Water early in the morning to allow foliage to dry out by evening, reducing the risk of mildew growth.",
            "### 4. Apply Organic Fungicides\nOrganic fungicides such as **neem oil**, **baking soda (sodium bicarbonate)**, or **sulfur** are effective against powdery mildew. These treatments can help to suppress the fungus while being safe for plants and the environment.",
            "### 5. Use Chemical Fungicides\nFor severe infestations, chemical fungicides containing active ingredients like **myclobutanil**, **tebuconazole**, or **chlorothalonil** can be used. Always follow the manufacturer's instructions and apply during early stages of infection for best results.",
            "### 6. Use a Milk Solution\nA simple homemade solution of **1 part milk** to **9 parts water** has been shown to effectively control powdery mildew. Spray the solution on affected areas every 7-10 days until the infection subsides.",
            "### 7. Maintain Plant Health\nHealthy plants are less susceptible to disease. Ensure that plants receive adequate sunlight, water, and nutrients. Over-fertilization, especially with nitrogen, can make plants more vulnerable to fungal diseases.",
            "### 8. Mulch Around Plants\nApply organic mulch around the base of plants to help regulate soil moisture, keep the roots cool, and reduce the spread of spores from the soil. Avoid excessive moisture buildup, as it can create an environment conducive to mildew growth.",
            "### 9. Use Resistant Varieties\nWhenever possible, choose plant varieties that are resistant to powdery mildew. Many varieties of vegetables, fruits, and ornamentals have been bred for resistance to this common fungal disease.",
            "### 10. Apply Horticultural Oils\nHorticultural oils, such as **summer oil** or **mineral oil**, can be sprayed on infected plants to suffocate the fungal spores. These oils work by disrupting the cell walls of the fungus, preventing it from spreading."
        ]
        
        # Display remedies
         for remedy in remedies:
            st.markdown(remedy)
        
        # Footer
         st.write("**Note**: Early detection and regular monitoring are key to preventing the spread of powdery mildew. Integrated pest management (IPM) using a combination of cultural, biological, and chemical methods will help protect your plants.")
                     
    elif labels1[zz[0][0]] == 8:
         
         print("----------------------------------------")
         print("Identified as Disease - SOOTY MOULD")
         print("----------------------------------------")    
         st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as Disease - SOOTY MOULD"}</h1>', unsafe_allow_html=True)

            
        # Introduction
         st.write("""
        Sooty Mold is a fungal disease that grows on the honeydew secreted by sap-sucking insects like aphids, scale, whiteflies, and mealybugs. While the mold itself doesn't directly harm the plant, it can block sunlight and reduce photosynthesis, leading to plant decline. Below are **10 effective remedies** to help manage and control sooty mold.
        """)
        
        # List of Remedies
         remedies = [
            "### 1. Control the Insect Infestation\nSince sooty mold grows on the honeydew produced by sap-sucking insects, controlling the insect population is the first step in preventing the mold. Use insecticidal soaps, neem oil, or chemical insecticides to target aphids, scale, whiteflies, and mealybugs.",
            "### 2. Wash the Plant Foliage\nAfter treating the pests, wash the plant leaves with a gentle spray of water to remove the sooty mold. This can help reduce the appearance of the mold and prevent it from suffocating the plant by blocking sunlight.",
            "### 3. Prune Infected Plant Parts\nPrune and remove heavily infected branches or leaves that have extensive sooty mold growth. Dispose of these parts properly to avoid spreading the fungal spores to healthy areas of the plant.",
            "### 4. Use Organic Fungicides\nApply organic fungicides such as **neem oil** or **horticultural oils** (e.g., summer oil). These can help control sooty mold by suffocating the fungal spores and preventing further spread.",
            "### 5. Apply Baking Soda Solution\nA simple solution of **1 tablespoon of baking soda** in **1 gallon of water** can act as a mild fungicide for sooty mold. Spray this solution on the affected plant surfaces to reduce mold growth.",
            "### 6. Increase Air Circulation\nEnsure that plants are spaced properly to allow air circulation, as high humidity and stagnant air encourage mold growth. Proper airflow helps the plant dry out more quickly after rain or watering, reducing the conditions for mold to thrive.",
            "### 7. Maintain Plant Health\nHealthy plants are more resilient to pests and diseases. Ensure that your plants are getting adequate sunlight, water, and nutrients. A well-nourished plant will be better able to withstand pest attacks and fungal infections like sooty mold.",
            "### 8. Introduce Natural Predators\nIntroduce natural predators like **ladybugs**, **lacewing larvae**, or **parasitic wasps** to control aphids, whiteflies, and other sap-sucking insects. Biological control can help manage the insect population and reduce the occurrence of sooty mold.",
            "### 9. Use Insecticidal Soap or Horticultural Oil\nInsecticidal soap or horticultural oils can help target and reduce the insect population that is creating the honeydew that leads to sooty mold. These products work by suffocating the pests, preventing further infestation.",
            "### 10. Use Sticky Traps for Monitoring\nPlace yellow sticky traps around your plants to monitor for the presence of flying insects such as whiteflies or aphids. These traps can help you keep track of pest levels and prevent future sooty mold outbreaks."
        ]
        
        # Display remedies
         for remedy in remedies:
            st.markdown(remedy)
        
        # Footer
         st.write("**Note**: Managing both the pests and the fungal mold is key to preventing the spread of sooty mold. Integrated pest management (IPM) using a combination of cultural, biological, and chemical methods will help keep your plants healthy.")
            
            
            
    
    
    
    
    
    
    
    
    
    
    