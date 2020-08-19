import pandas as pd
import patsy as ps
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt


#read the data
listing = pd.read_csv('listings.csv')

#print the data
listing.columns

#convert data into Panda's Dataframe
df = listing[['host_acceptance_rate', 'host_is_superhost',
       'host_neighbourhood','host_listings_count', 'host_has_profile_pic', 'host_identity_verified',
       'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
       'smart_location', 'country_code', 'country', 'latitude', 'longitude',
       'is_location_exact', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'square_feet',
       'price',  'security_deposit',
       'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
       'maximum_nights', 'has_availability',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365', 'number_of_reviews', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value',
        'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture',
       'require_guest_phone_verification', 'calculated_host_listings_count',
       'reviews_per_month']]



#What are the dimensions of the dataset?
num_rows = df.shape[0] #Provide the number of rows in the dataset
num_cols = df.shape[1] #Provide the number of columns in the dataset


#Data Cleaning:
df = df.dropna(subset = ['price'], how='any')
df["price"]=df.price.str.replace('$','').str.replace(',','').astype(float)
df["security_deposit"]=df.security_deposit.str.replace('$','').str.replace(',','').astype(float)
df["cleaning_fee"]=df.cleaning_fee.str.replace('$','').str.replace(',','').astype(float)
df["host_acceptance_rate"]=df.host_acceptance_rate.str.replace('%','').str.replace(',','').astype(float)
df["extra_people"]=df.extra_people.str.replace('$','').str.replace(',','').astype(float)
df['host_acceptance_rate'].fillna(df.host_acceptance_rate.mean(), inplace=True)
df['host_neighbourhood'].fillna("unknown", inplace=True)
df['square_feet'].fillna(df.square_feet.mean(), inplace=True)
df['security_deposit'].fillna(df.security_deposit.mean(), inplace=True)
df['cleaning_fee'].fillna(df.cleaning_fee.mean(), inplace=True)
df['review_scores_rating'].fillna(df.review_scores_rating.mean(), inplace=True)
df['review_scores_accuracy'].fillna(df.review_scores_accuracy.mean(), inplace=True)
df['review_scores_cleanliness'].fillna(df.review_scores_cleanliness.mean(), inplace=True)
df['review_scores_checkin'].fillna(df.review_scores_checkin.mean(), inplace=True)
df['review_scores_communication'].fillna(df.review_scores_communication.mean(), inplace=True)
df['review_scores_location'].fillna(df.review_scores_location.mean(), inplace=True)
df['review_scores_value'].fillna(df.review_scores_value.mean(), inplace=True)
df['review_scores_value'].fillna(df.review_scores_value.mean(), inplace=True)
df['reviews_per_month'].fillna(0, inplace=True)



#Make the linear model, vars are Proerty Type, Room Type, Bed Type, Zip Code, Nr of People, Nr of Bathrooms, Nr of Bedrooms, Nr of Beds, Bed Type, Nr of Guests in the Reservation, Extra People
y, X = ps.dmatrices("price ~ 1 +  C(property_type)+ C(room_type)+ C(bed_type)+ C(zipcode)+"
                    "  accommodates + bathrooms + bedrooms + beds + bed_type +  "
                    " guests_included + extra_people ",df, return_type='dataframe')

#Split the data into a train and test set, test set is 30% of total dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)

#Choose the model characteristics, eg linear, quadratic or other
lm_model = LinearRegression(normalize=False)  # Instantiate

#Fit the model to the training data (70% of the dataset)
lm_model.fit(X_train.values, y_train.values)  # Fit

# Predict and score the model
y_test_preds = lm_model.predict(X_test.values)

# Rsquared metric to see how well the model performs on unseen data
rsquared_score = r2_score(y_test.values, y_test_preds)
print("R-squared of model is:",rsquared_score)



#Extract the coefficients of the linear model
coeff_df = pd.Series(lm_model.coef_.ravel(), index=X.columns)
print(coeff_df)


# graphs


df.price.hist(bins=50)
df.neighbourhood_group_cleansed.hist(bins=1000, fontsize=5)
price_neighbourhood = df.groupby(['neighbourhood_group_cleansed']).agg({'price':'mean'}).reset_index()
plt.scatter(price_neighbourhood['neighbourhood_group_cleansed'], price_neighbourhood['price'])



# actial labels

plt.bar(price_neighbourhood['neighbourhood_group_cleansed'], price_neighbourhood['price'])

price_neighbourhood.plot.bar(rot=40,figsize=[15,10],fontsize=5)

plt.hexbin(y_test.values, y_test_preds, bins='log')

# regression plot

plt.hist(y_test.values - y_test_preds, bins=50)
np.std(y_test.values - y_test_preds)
