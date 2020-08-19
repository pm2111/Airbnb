import pandas as pd
import patsy as ps
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt



listing = pd.read_csv('listings.csv')


listing.columns


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


#df = pd.merge(cal, list_chosen, how='inner', left_on='listing_id', right_on = 'id')

num_rows = df.shape[0] #Provide the number of rows in the dataset
num_cols = df.shape[1] #Provide the number of columns in the dataset


#df.drop(['id', 'listing_id', 'available'], axis=1, inplace=True)
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




y, X = ps.dmatrices("price ~ 1 + + C(property_type)+ C(room_type)+ C(bed_type)+ C(zipcode)+"
                    "  accommodates + bathrooms + bedrooms + beds + bed_type +  "
                    " guests_included + extra_people ",df, return_type='dataframe')

'''
X = df[['host_acceptance_rate', 'host_is_superhost',
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

y = df['price']


C(host_has_profile_pic)+ C(host_is_superhost)+ C(host_neighbourhood)+ C(host_has_profile_pic)+ C(host_identity_verified)+ C(neighbourhood_group_cleansed)+ "
                    "C(smart_location)+ C(country_code)+ C(country)+ C(is_location_exact)
C(has_availability) + C(instant_bookable)+ C(cancellation_policy)+ "
                    "C(require_guest_profile_picture) + C(require_guest_phone_verification)

host_acceptance_rate + host_listings_count + latitude + longitude

security_deposit + review_scores_location + review_scores_value + 


'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)

lm_model = LinearRegression(normalize=False)  # Instantiate
lm_model.fit(X_train.values, y_train.values)  # Fit

# Predict and score the model
y_test_preds = lm_model.predict(X_test.values)

# Rsquared and y_test
rsquared_score = r2_score(y_test, y_test_preds)
length_y_test = len(y_test)



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