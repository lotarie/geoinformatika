import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import rasterio 

#B1 - ndvi, b2 - blue, b3 - green, b4 - red, b5 - nir
#1-water, 2-forest, 3- urban area
sat_img_all_bands = "C://Users//karol//Desktop//MAGISTR//GEOINFROMATIKA//ukol_machine//qgis//rastr_all_bands.tif"       
train_polygons = "C://Users//karol//Desktop//MAGISTR//GEOINFROMATIKA//ukol_machine//qgis//train_data_raster.tif" 
test_polygons = "C://Users//karol//Desktop//MAGISTR//GEOINFROMATIKA//ukol_machine//qgis//test_data_raster.tif"


attributes = ["NDVI", "Blue", "Green", "Red", "NIR"]



###############################
#X - data 
with rasterio.open(sat_img_all_bands) as src_img:

    img_data = src_img.read() 
    img_height = src_img.height
    img_width = src_img.width
    num_of_bands = src_img.count
    print(f"size of satellite image: {img_data.shape} (bands, height, width)")


#y train
with rasterio.open(train_polygons) as src_mask:
    mask_data = src_mask.read(1)
    print(f"size of mask: {mask_data.shape}")
    
#y test
with rasterio.open(test_polygons) as src_mask_test:
    mask_data_test = src_mask_test.read(1)
    print(f"size of test mask: {mask_data_test.shape}")
    
#check sizes    
if img_data.shape[1:] != mask_data.shape:
    raise ValueError(f"sizes are different. Size of satellite data {img_data.shape[1:]}, size of mask {mask_data.shape}.")



#reshape data for ML
X_all = img_data.reshape(num_of_bands, -1).T  # (num_of_pixels, num_of_bands)
y_train_all = mask_data.reshape(-1)               # (num_of_pixels, )
y_test_all = mask_data_test.reshape(-1)      # (num_of_pixels, )


# pixels with value > 0 (we want to train only on labeled pixels)
pixels_with_value = y_train_all > 0

# final training data
X_train = X_all[pixels_with_value]
y_train = y_train_all[pixels_with_value]

# create dataframe for better visualization
df_train = pd.DataFrame(X_train, columns=attributes)
df_train['Label'] = y_train


#######################
#plot training data
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
data_for_plotting = df_train.sample(n=min(2000, len(df_train)))
scatter = sns.scatterplot(
    data=data_for_plotting,
    x="Red",        
    y="NIR",          
    hue="Label",     
    palette="bright", 
    s=50)

#1-water, 2-forest, 3-urban area
plt.title("Scatterplot of spectral indices: Red vs. NIR", fontsize=15)
plt.xlabel("Reflectance in (Red)", fontsize=12)
plt.ylabel("Reflectance in (NIR)", fontsize=12)
plt.legend(title="Land cover class")
plt.show()


##############################
#train model 

#prepare test data
pixels_with_value_test = y_test_all > 0
X_test = X_all[pixels_with_value_test]
y_test = y_test_all[pixels_with_value_test]



# test different tree depths
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25] 
results = []

print(f"{'Depth':<10} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
print("-" * 45)

for d in depths:

    model = DecisionTreeClassifier(max_depth=d, min_samples_leaf=50)
    model.fit(X_train, y_train)

    y_hat_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_hat_train)

    y_hat_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_hat_test)

    d_str = str(d) if d is not None else "Max"
    print(f"{d_str:<10} | {acc_train:.4f}          | {acc_test:.4f}")
    
    results.append({'depth': d_str, 'train_acc': acc_train, 'test_acc': acc_test})













