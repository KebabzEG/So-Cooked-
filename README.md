
## Cuisine Prediction from Ingredients

Problem statement: How can we classify the category of cuisine given a list of ingredients from a recipe?

Objective: Implement a prediction system that uses recipe-ingredient data from Kaggle challenge (https://www.kaggle.com/competitions/whats-cooking/data) to classify the category of cuisine of a given recipe





```python
# Import the libraries

import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

```

## Extract the data from the train.json file

Create a DataFrame to store the file data


```python
# Read data from train.json file
recipe_data = pd.read_json('data/train.json')
recipe_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
    </tr>
  </tbody>
</table>
</div>



## Inspect the attributes of the DataFrames

Print out the sizes and data types of the DataFrames, checking for null values as well


```python
# Check data type
print("Data type:", type(recipe_data))

# Check size of dataset
print('Data dim:', recipe_data.shape) 
print('Columns:', recipe_data.columns)

# Check variable types in dataset
print('\n', recipe_data.dtypes)

# Checking for null values in the data
print('\nWhether Null exists:\n', recipe_data.isnull().sum())
```

    Data type: <class 'pandas.core.frame.DataFrame'>
    Data dim: (39774, 3)
    Columns: Index(['id', 'cuisine', 'ingredients'], dtype='object')
    
     id              int64
    cuisine        object
    ingredients    object
    dtype: object
    
    Whether Null exists:
     id             0
    cuisine        0
    ingredients    0
    dtype: int64
    

## Lemmatization

Reduce words to their basic form to reduce variations in what would essentially be the same ingredient (i.e., tomatoes and tomato, baked and bake).

The ingredients are further reduced by removing duplicates, only leaving unique ingredients


```python
# Lemmatization to reduce words to base form i.e. tomatoes to tomato
lemmatizer = WordNetLemmatizer()

# Convert lowercase for all recipe ingredients
recipe_data['ingredients_lower'] = recipe_data['ingredients'].apply(
    lambda ingredients: [item.lower() for item in ingredients]
)
recipe_data['ingredients_lem'] = recipe_data['ingredients_lower'].apply(
    lambda ingredients: [lemmatizer.lemmatize(item) for item in ingredients]
)

# Convert list of ingredients to string, separated by comma
recipe_data['ingredients_str'] = recipe_data['ingredients_lem'].apply(lambda x: ', '.join(x))

# Check for duplicates based on the 'id' column
duplicate_ids = recipe_data.duplicated(subset=['id']).sum()
print("No. of Duplicate IDs:\n", duplicate_ids)

# Check for duplicates based on 'cuisine' and 'ingredients_str' column
duplicate_rows = recipe_data.duplicated(subset=['cuisine', 'ingredients_str']).sum()
print("No. of duplicate rows:\n", duplicate_rows)

# Drop duplicate rows based on 'cuisine' and 'ingredients_str' columns, keeping the first occurrence
recipe_data = recipe_data.drop_duplicates(subset=['cuisine', 'ingredients_str'], keep='first')
# Reset index after dropping duplicates
recipe_data = recipe_data.reset_index(drop=True)
# Verify duplicate rows have been dropped
duplicate_rows = recipe_data.duplicated(subset=['cuisine', 'ingredients_str']).sum()
print("Remaining no. of duplicate rows:\n", duplicate_rows)

print('New data dim:', recipe_data.shape) 
recipe_data.head(n = 10)
```

    No. of Duplicate IDs:
     0
    No. of duplicate rows:
     97
    Remaining no. of duplicate rows:
     0
    New data dim: (39677, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
      <th>ingredients_lower</th>
      <th>ingredients_lem</th>
      <th>ingredients_str</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>romaine lettuce, black olives, grape tomatoes,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>[plain flour, ground pepper, salt, tomato, gro...</td>
      <td>plain flour, ground pepper, salt, tomato, grou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>[egg, pepper, salt, mayonaise, cooking oil, gr...</td>
      <td>egg, pepper, salt, mayonaise, cooking oil, gre...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>water, vegetable oil, wheat, salt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
      <td>[black pepper, shallot, cornflour, cayenne pep...</td>
      <td>black pepper, shallot, cornflour, cayenne pepp...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6602</td>
      <td>jamaican</td>
      <td>[plain flour, sugar, butter, eggs, fresh ginge...</td>
      <td>[plain flour, sugar, butter, eggs, fresh ginge...</td>
      <td>[plain flour, sugar, butter, egg, fresh ginger...</td>
      <td>plain flour, sugar, butter, egg, fresh ginger ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>42779</td>
      <td>spanish</td>
      <td>[olive oil, salt, medium shrimp, pepper, garli...</td>
      <td>[olive oil, salt, medium shrimp, pepper, garli...</td>
      <td>[olive oil, salt, medium shrimp, pepper, garli...</td>
      <td>olive oil, salt, medium shrimp, pepper, garlic...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3735</td>
      <td>italian</td>
      <td>[sugar, pistachio nuts, white almond bark, flo...</td>
      <td>[sugar, pistachio nuts, white almond bark, flo...</td>
      <td>[sugar, pistachio nuts, white almond bark, flo...</td>
      <td>sugar, pistachio nuts, white almond bark, flou...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16903</td>
      <td>mexican</td>
      <td>[olive oil, purple onion, fresh pineapple, por...</td>
      <td>[olive oil, purple onion, fresh pineapple, por...</td>
      <td>[olive oil, purple onion, fresh pineapple, por...</td>
      <td>olive oil, purple onion, fresh pineapple, pork...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12734</td>
      <td>italian</td>
      <td>[chopped tomatoes, fresh basil, garlic, extra-...</td>
      <td>[chopped tomatoes, fresh basil, garlic, extra-...</td>
      <td>[chopped tomatoes, fresh basil, garlic, extra-...</td>
      <td>chopped tomatoes, fresh basil, garlic, extra-v...</td>
    </tr>
  </tbody>
</table>
</div>



## View the attributes of this cleaned DataFrame

Methods used include .nunique(), .unique() and .value_counts() 


```python
# Find how many unique cuisines are there
print(recipe_data['cuisine'].nunique())

# What are the unique cuisines?
print(recipe_data['cuisine'].unique())
```

    20
    ['greek' 'southern_us' 'filipino' 'indian' 'jamaican' 'spanish' 'italian'
     'mexican' 'chinese' 'british' 'thai' 'vietnamese' 'cajun_creole'
     'brazilian' 'french' 'japanese' 'irish' 'korean' 'moroccan' 'russian']
    


```python
# Find how many recipes are there
recipe_data['ingredients_lem']
```




    0        [romaine lettuce, black olives, grape tomatoes...
    1        [plain flour, ground pepper, salt, tomato, gro...
    2        [egg, pepper, salt, mayonaise, cooking oil, gr...
    3                      [water, vegetable oil, wheat, salt]
    4        [black pepper, shallot, cornflour, cayenne pep...
                                   ...                        
    39672    [light brown sugar, granulated sugar, butter, ...
    39673    [kraft zesty italian dressing, purple onion, b...
    39674    [egg, citrus fruit, raisin, sourdough starter,...
    39675    [boneless chicken skinless thigh, minced garli...
    39676    [green chile, jalapeno chilies, onion, ground ...
    Name: ingredients_lem, Length: 39677, dtype: object




```python
# Example of a single recipe column
recipe_data['ingredients_lem'][4]
```




    ['black pepper',
     'shallot',
     'cornflour',
     'cayenne pepper',
     'onion',
     'garlic paste',
     'milk',
     'butter',
     'salt',
     'lemon juice',
     'water',
     'chili powder',
     'passata',
     'oil',
     'ground cumin',
     'boneless chicken skinless thigh',
     'garam masala',
     'double cream',
     'natural yogurt',
     'bay leaf']




```python
# Count recipes for each cuisine
recipe_data['cuisine'].value_counts()
```




    cuisine
    italian         7831
    mexican         6429
    southern_us     4299
    indian          2997
    chinese         2666
    french          2637
    cajun_creole    1541
    thai            1536
    japanese        1417
    greek           1172
    spanish          987
    korean           827
    vietnamese       821
    moroccan         818
    british          803
    filipino         755
    irish            667
    jamaican         522
    russian          489
    brazilian        463
    Name: count, dtype: int64



## Data Visualisation

Visualise the number of recipes each cuisine has with each other using statistical plots like countplot, histograms, pie charts and WordCloud Generator


```python
# Visualise on count plot

plt.figure(figsize=(24, 4))  # Adjust figure size as needed
sb.countplot(data=recipe_data, x='cuisine', order=recipe_data['cuisine'].value_counts().index)
plt.xticks(rotation=90)  # Rotate x-axis labels if needed for readability
plt.title('Number of Recipes by Cuisine')
plt.xlabel('Cuisine')
plt.ylabel('Number of Recipes')
plt.show()
```


    
![png](output_14_0.png)
    



```python
# Visualise cuisine distribution on a bar graph
def plot_cuisine_distribution_bar():
    plt.figure(figsize=(12, 6))
    sb.countplot(data=recipe_data, x='cuisine', order=recipe_data['cuisine'].value_counts().index, palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.title('Cuisine Distribution (Bar Plot)')
    plt.xlabel('Cuisine')
    plt.ylabel('Count')
    plt.show()

plot_cuisine_distribution_bar()
```


    
![png](output_15_0.png)
    



```python
# Visualise cuisine distribution of dataset on a pie chart
def plot_cuisine_distribution_pie():
    plt.figure(figsize=(8, 8))
    fig = recipe_data['cuisine'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set3', pctdistance = 0.8)
    plt.setp(fig.texts, size = 4.5)
    plt.title('Cuisine Distribution (Pie Chart)')
    plt.ylabel('')
    plt.show()

plot_cuisine_distribution_pie()
```


    
![png](output_16_0.png)
    


## WordCloud Generator

Generate a graphic with words of cuisines varying sizes that correspond with the number of recipes. The bigger the cuisine word, the more recipes it has


```python
# Install wordcloud library
%pip install wordcloud

# Generate wordcloud, largest size -> most number of recipes
from wordcloud import WordCloud

x = recipe_data['cuisine'].values

plt.subplots(figsize = (8,8))

wordcloud = WordCloud (
                    background_color = 'white',
                    width = 712,
                    height = 384,
                    colormap = 'prism').generate(' '.join(x))
plt.imshow(wordcloud) # image show
plt.axis('off') # to off the axis of x and y
plt.savefig('cuisines.png')
plt.show()
```

    Requirement already satisfied: wordcloud in c:\users\rayson\anaconda3\lib\site-packages (1.9.3)
    Requirement already satisfied: numpy>=1.6.1 in c:\users\rayson\anaconda3\lib\site-packages (from wordcloud) (1.26.4)
    Requirement already satisfied: pillow in c:\users\rayson\anaconda3\lib\site-packages (from wordcloud) (10.2.0)
    Requirement already satisfied: matplotlib in c:\users\rayson\anaconda3\lib\site-packages (from wordcloud) (3.8.0)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\rayson\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\rayson\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\rayson\anaconda3\lib\site-packages (from matplotlib->wordcloud) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\rayson\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in c:\users\rayson\anaconda3\lib\site-packages (from matplotlib->wordcloud) (23.1)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\rayson\anaconda3\lib\site-packages (from matplotlib->wordcloud) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\rayson\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\rayson\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


    
![png](output_18_1.png)
    


## Ingredient Sorting

Visualise number of each ingredient with plottings


```python
# Visualise number of ingredients for each recipe
recipe_data['num_ingredients'] = recipe_data['ingredients_lem'].apply(lambda x: len(x))

plt.figure(figsize=(20,4))
sb.countplot(x='num_ingredients', data=recipe_data)
plt.show()
```


    
![png](output_20_0.png)
    



```python
recipe_data_exploded = recipe_data.explode('ingredients_lem')

# How many unique ingredients are there?
print("Total no. of unique ingredients:", recipe_data_exploded['ingredients_lem'].nunique())
recipe_data_exploded_sorted = recipe_data_exploded.sort_values(by='ingredients_lem').reset_index(drop=True)
print("Unique ingredients:", recipe_data_exploded_sorted['ingredients_lem'].unique()[:200])
```

    Total no. of unique ingredients: 6703
    Unique ingredients: ['(    oz.) tomato sauce' '(   oz.) tomato paste'
     '(10 oz.) frozen chopped spinach'
     '(10 oz.) frozen chopped spinach, thawed and squeezed dry'
     '(14 oz.) sweetened condensed milk' '(14.5 oz.) diced tomatoes'
     '(15 oz.) refried beans' '1% low-fat buttermilk'
     '1% low-fat chocolate milk' '1% low-fat cottage cheese' '1% low-fat milk'
     '2 1/2 to 3 lb. chicken, cut into serving pieces'
     '2% low fat cheddar chees' '2% low-fat cottage cheese'
     '2% lowfat greek yogurt' '2% milk shredded mozzarella cheese'
     '2% reduced-fat milk' '25% less sodium chicken broth'
     '33% less sodium cooked deli ham' '33% less sodium cooked ham'
     '33% less sodium ham' '33% less sodium smoked fully cooked ham'
     '40% less sodium taco seasoning' '40% less sodium taco seasoning mix'
     '7 up' '8 ounc ziti pasta, cook and drain' '95% lean ground beef'
     'a taste of thai rice noodles' 'abalone' 'abbamele' 'absinthe'
     'abura age' 'acai juice' 'accent' 'accent seasoning' 'accompaniment'
     'achiote' 'achiote paste' 'achiote powder' 'acini di pepe' 'ackee'
     'acorn squash' 'active dry yeast' 'adobo' 'adobo all purpose seasoning'
     'adobo sauce' 'adobo seasoning' 'adobo style seasoning' 'adzuki beans'
     'agar' 'agar agar flakes' 'agave nectar' 'agave tequila'
     'aged balsamic vinegar' 'aged cheddar cheese' 'aged gouda'
     'aged manchego cheese' 'ahi' 'ahi tuna steaks' 'aioli' 'ajinomoto'
     'ajwain' 'aka miso' 'alaskan king crab legs' 'alaskan king salmon'
     'albacore' 'albacore tuna in water' 'alcohol' 'ale' 'aleppo'
     'aleppo pepper' 'alexia waffle fries' 'alfalfa sprouts' 'alfredo sauce'
     'alfredo sauce mix' 'all beef hot dogs' 'all potato purpos'
     'all purpose seasoning' 'all purpose unbleached flour'
     'all-purpose flour' 'allspice' 'allspice berries' 'almond'
     'almond butter' 'almond extract' 'almond filling' 'almond flour'
     'almond liqueur' 'almond meal' 'almond milk' 'almond oil' 'almond paste'
     'almond syrup' 'aloe juice' 'alphabet pasta' 'alum' 'amaranth'
     'amarena cherries' 'amaretti' 'amaretti cookies' 'amaretto'
     'amaretto liqueur' 'amba' 'amber' 'amber rum' 'amberjack fillet' 'amchur'
     'america' 'american cheese' 'american cheese food'
     'american cheese slices' 'ammonium bicarbonate' 'amontillado sherry'
     'ampalaya' 'anaheim chile' 'anasazi beans' 'ancho' 'ancho chile pepper'
     'ancho chili ground pepper' 'ancho powder' 'anchovy' 'anchovy filets'
     'anchovy fillets' 'anchovy paste' 'and carrot green pea'
     'and cook drain pasta ziti' 'and fat free half half'
     'andouille chicken sausage' 'andouille sausage' 'andouille sausage links'
     'andouille turkey sausages' 'angel food cake' 'angel food cake mix'
     'angel hair' 'angled loofah' 'angostura bitters' 'angus' 'anise'
     'anise basil' 'anise extract' 'anise liqueur' 'anise oil' 'anise powder'
     'anise seed' 'anisette' 'anjou pears' 'annatto' 'annatto oil'
     'annatto powder' 'annatto seeds' 'any' 'aonori' 'apple' 'apple brandy'
     'apple butter' 'apple cider' 'apple cider vinegar' 'apple jelly'
     'apple juice' 'apple juice concentrate' 'apple pie filling'
     'apple pie spice' 'apple puree' 'apple schnapps' 'apple slice'
     'applesauce' 'applewood smoked bacon' 'apricot' 'apricot brandy'
     'apricot halves' 'apricot jam' 'apricot jelly' 'apricot nectar'
     'apricot preserves' 'aquavit' 'arak' 'arame' 'arbol chile' 'arborio rice'
     'arctic char' 'arepa flour' 'argo corn starch' 'arhar' 'arhar dal'
     'armagnac' 'arrow root' 'arrowroot' 'arrowroot flour' 'arrowroot powder'
     'arrowroot starch' 'artichok heart marin' 'artichoke' 'artichoke bottoms'
     'artichoke hearts' 'artificial sweetener' 'artisan bread' 'arugula'
     'asadero' 'asafetida' 'asafetida (powder)']
    


```python
# Get top 10 ingredients of cuisine
def plot_top_ingredients(cuisine, top_n=10):
    top_ingredients = (recipe_data_exploded[recipe_data_exploded['cuisine'] == cuisine].value_counts('ingredients_lem').head(top_n))

    plt.figure(figsize=(10, 6))
    sb.barplot(x=top_ingredients.values, y=top_ingredients.index, palette='viridis')
    plt.title(f"Top {top_n} Ingredients in {cuisine.capitalize()} Cuisine")
    plt.xlabel("Frequency")
    plt.ylabel("Ingredient")
    plt.show()

# Visualise top 10 ingredients for italian cuisine
plot_top_ingredients('italian')
```


    
![png](output_22_0.png)
    



```python
# Visualise top 10 ingredients for mexican cuisine
plot_top_ingredients('mexican')
```


    
![png](output_23_0.png)
    



```python
# Visualise top 10 ingredients for brazil cuisine
plot_top_ingredients('brazilian')
```


    
![png](output_24_0.png)
    


## Remove ingredients that are common across many cuisines



From the bar graphs of the top 10 ingredients in various cuisines, we found out that there were common ingredients e.g. water, salt, sugar, oil across the cuisine types. As these could introduce noise when applying machine learning tools, we decided to remove them


```python
# List of common ingredients to remove
common_ingredients = ["water", "sugar", "salt", "pepper", "black pepper", "ground pepper", "ground black pepper",
                         "shallot", "vegetable oil", "cooking oil", "olive oil", "extra-virgin olive oil",
                         "condiment", "seasoning", "onion", "purple onion", "yellow onion", "garlic", "garlic cloves", "butter"]

# Apply the function to clean the 'ingredients_lem' column
recipe_data['ingredients_lem'] = recipe_data['ingredients_lem'].apply(
     lambda ingredients: [item for item in ingredients if item not in common_ingredients]
)

# Update the 'ingredients_str' column based on the 'ingredients_lem' list
recipe_data['ingredients_str'] = recipe_data['ingredients_lem'].apply(lambda x: ', '.join(x))

# Recalculate num_ingredients after cleaning
recipe_data['num_ingredients'] = recipe_data['ingredients_lem'].apply(len)

# Display the updated DataFrame to confirm cleaning
recipe_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
      <th>ingredients_lower</th>
      <th>ingredients_lem</th>
      <th>ingredients_str</th>
      <th>num_ingredients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>romaine lettuce, black olives, grape tomatoes,...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>[plain flour, tomato, thyme, egg, green tomato...</td>
      <td>plain flour, tomato, thyme, egg, green tomatoe...</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>[egg, mayonaise, green chilies, grilled chicke...</td>
      <td>egg, mayonaise, green chilies, grilled chicken...</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>[wheat]</td>
      <td>wheat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
      <td>[cornflour, cayenne pepper, garlic paste, milk...</td>
      <td>cornflour, cayenne pepper, garlic paste, milk,...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6602</td>
      <td>jamaican</td>
      <td>[plain flour, sugar, butter, eggs, fresh ginge...</td>
      <td>[plain flour, sugar, butter, eggs, fresh ginge...</td>
      <td>[plain flour, egg, fresh ginger root, ground c...</td>
      <td>plain flour, egg, fresh ginger root, ground ci...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>42779</td>
      <td>spanish</td>
      <td>[olive oil, salt, medium shrimp, pepper, garli...</td>
      <td>[olive oil, salt, medium shrimp, pepper, garli...</td>
      <td>[medium shrimp, chopped cilantro, jalapeno chi...</td>
      <td>medium shrimp, chopped cilantro, jalapeno chil...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3735</td>
      <td>italian</td>
      <td>[sugar, pistachio nuts, white almond bark, flo...</td>
      <td>[sugar, pistachio nuts, white almond bark, flo...</td>
      <td>[pistachio nuts, white almond bark, flour, van...</td>
      <td>pistachio nuts, white almond bark, flour, vani...</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16903</td>
      <td>mexican</td>
      <td>[olive oil, purple onion, fresh pineapple, por...</td>
      <td>[olive oil, purple onion, fresh pineapple, por...</td>
      <td>[fresh pineapple, pork, poblano peppers, corn ...</td>
      <td>fresh pineapple, pork, poblano peppers, corn t...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12734</td>
      <td>italian</td>
      <td>[chopped tomatoes, fresh basil, garlic, extra-...</td>
      <td>[chopped tomatoes, fresh basil, garlic, extra-...</td>
      <td>[chopped tomatoes, fresh basil, kosher salt, f...</td>
      <td>chopped tomatoes, fresh basil, kosher salt, fl...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
recipe_data_exploded = recipe_data.explode('ingredients_lem')

# Visualise top 10 ingredients for italian cuisine
plot_top_ingredients('italian')
```


    
![png](output_28_0.png)
    



```python
# Visualise top 10 ingredients for mexican cuisine
plot_top_ingredients('mexican')
```


    
![png](output_29_0.png)
    



```python
# Visualise top 10 ingredients for Southern US cuisine
plot_top_ingredients('brazilian')
```


    
![png](output_30_0.png)
    


## Group the data

Transform the dataset to be an array with all the neccessary ingredients. Utilise a CountVectoriser() object to count the ingredient strings using a Bag-Of-Words (BoW) model


```python
# Transform the dataset so that each row represents a cuisine and contains a single string with all the ingredients used across recipes of that cuisine
cuisine_ingredients = recipe_data.groupby('cuisine')['ingredients_str'].apply(lambda x: ' '.join(x)).reset_index()



# To convert the textual data of ingredients into a numerical format using a bag-of-words model
vectorizer = CountVectorizer()
ingredient_matrix = vectorizer.fit_transform(cuisine_ingredients['ingredients_str']).toarray()

# Show the vector size
number_of_cuisines, vector_size = ingredient_matrix.shape

print("Number of cuisines:",number_of_cuisines)
print("Vector size:",vector_size)
print(vectorizer.get_feature_names_out())
```

    Number of cuisines: 20
    Vector size: 3002
    ['00' '10' '100' ... 'ziti' 'zucchini' 'épices']
    

## Create a DataFrame from the matrix  

Each row represents a cuisine, and each row a unique ingredient. Each cell corresponds to a count of that ingredient for said cuisine.

This format makes it easier to visualise and compute relationships between cuisines and their shared ingredients.

Thus, we calculate the correlation between them. The correlation values range from -1 to 1


```python
# Create a DataFrame for ingredient matrix with cuisines as index
ingredient_df = pd.DataFrame(ingredient_matrix, index=cuisine_ingredients['cuisine'], columns=vectorizer.get_feature_names_out())



# Calculate correlation matrix between cuisines
cuisine_correlation = ingredient_df.T.corr()  # Transpose to get correlation across cuisines
cuisine_correlation
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>cuisine</th>
      <th>brazilian</th>
      <th>british</th>
      <th>cajun_creole</th>
      <th>chinese</th>
      <th>filipino</th>
      <th>french</th>
      <th>greek</th>
      <th>indian</th>
      <th>irish</th>
      <th>italian</th>
      <th>jamaican</th>
      <th>japanese</th>
      <th>korean</th>
      <th>mexican</th>
      <th>moroccan</th>
      <th>russian</th>
      <th>southern_us</th>
      <th>spanish</th>
      <th>thai</th>
      <th>vietnamese</th>
    </tr>
    <tr>
      <th>cuisine</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>brazilian</th>
      <td>1.000000</td>
      <td>0.580831</td>
      <td>0.658474</td>
      <td>0.439457</td>
      <td>0.625740</td>
      <td>0.650146</td>
      <td>0.552962</td>
      <td>0.593421</td>
      <td>0.562573</td>
      <td>0.594306</td>
      <td>0.695495</td>
      <td>0.460317</td>
      <td>0.363876</td>
      <td>0.676835</td>
      <td>0.509188</td>
      <td>0.586674</td>
      <td>0.676681</td>
      <td>0.727687</td>
      <td>0.686447</td>
      <td>0.557734</td>
    </tr>
    <tr>
      <th>british</th>
      <td>0.580831</td>
      <td>1.000000</td>
      <td>0.461292</td>
      <td>0.330571</td>
      <td>0.495386</td>
      <td>0.817104</td>
      <td>0.470475</td>
      <td>0.427725</td>
      <td>0.905310</td>
      <td>0.536510</td>
      <td>0.558891</td>
      <td>0.395659</td>
      <td>0.276495</td>
      <td>0.451349</td>
      <td>0.410732</td>
      <td>0.883071</td>
      <td>0.875686</td>
      <td>0.503061</td>
      <td>0.338618</td>
      <td>0.321151</td>
    </tr>
    <tr>
      <th>cajun_creole</th>
      <td>0.658474</td>
      <td>0.461292</td>
      <td>1.000000</td>
      <td>0.519573</td>
      <td>0.579078</td>
      <td>0.584640</td>
      <td>0.560105</td>
      <td>0.486997</td>
      <td>0.499915</td>
      <td>0.620576</td>
      <td>0.664057</td>
      <td>0.473538</td>
      <td>0.419077</td>
      <td>0.630519</td>
      <td>0.493638</td>
      <td>0.516681</td>
      <td>0.675126</td>
      <td>0.774291</td>
      <td>0.543496</td>
      <td>0.501517</td>
    </tr>
    <tr>
      <th>chinese</th>
      <td>0.439457</td>
      <td>0.330571</td>
      <td>0.519573</td>
      <td>1.000000</td>
      <td>0.836513</td>
      <td>0.371484</td>
      <td>0.321137</td>
      <td>0.427245</td>
      <td>0.317347</td>
      <td>0.379484</td>
      <td>0.545463</td>
      <td>0.891298</td>
      <td>0.861580</td>
      <td>0.450431</td>
      <td>0.335313</td>
      <td>0.352506</td>
      <td>0.453500</td>
      <td>0.461079</td>
      <td>0.718632</td>
      <td>0.783786</td>
    </tr>
    <tr>
      <th>filipino</th>
      <td>0.625740</td>
      <td>0.495386</td>
      <td>0.579078</td>
      <td>0.836513</td>
      <td>1.000000</td>
      <td>0.474612</td>
      <td>0.361439</td>
      <td>0.458049</td>
      <td>0.472240</td>
      <td>0.431508</td>
      <td>0.622334</td>
      <td>0.790714</td>
      <td>0.691773</td>
      <td>0.504062</td>
      <td>0.359234</td>
      <td>0.504358</td>
      <td>0.581332</td>
      <td>0.512150</td>
      <td>0.736732</td>
      <td>0.771097</td>
    </tr>
    <tr>
      <th>french</th>
      <td>0.650146</td>
      <td>0.817104</td>
      <td>0.584640</td>
      <td>0.371484</td>
      <td>0.474612</td>
      <td>1.000000</td>
      <td>0.685636</td>
      <td>0.460841</td>
      <td>0.781788</td>
      <td>0.767632</td>
      <td>0.577509</td>
      <td>0.408776</td>
      <td>0.281242</td>
      <td>0.541181</td>
      <td>0.522960</td>
      <td>0.847700</td>
      <td>0.802111</td>
      <td>0.754748</td>
      <td>0.443445</td>
      <td>0.404030</td>
    </tr>
    <tr>
      <th>greek</th>
      <td>0.552962</td>
      <td>0.470475</td>
      <td>0.560105</td>
      <td>0.321137</td>
      <td>0.361439</td>
      <td>0.685636</td>
      <td>1.000000</td>
      <td>0.517263</td>
      <td>0.488824</td>
      <td>0.784062</td>
      <td>0.541020</td>
      <td>0.354222</td>
      <td>0.255248</td>
      <td>0.580871</td>
      <td>0.642514</td>
      <td>0.568039</td>
      <td>0.547778</td>
      <td>0.726851</td>
      <td>0.438834</td>
      <td>0.412951</td>
    </tr>
    <tr>
      <th>indian</th>
      <td>0.593421</td>
      <td>0.427725</td>
      <td>0.486997</td>
      <td>0.427245</td>
      <td>0.458049</td>
      <td>0.460841</td>
      <td>0.517263</td>
      <td>1.000000</td>
      <td>0.434393</td>
      <td>0.436468</td>
      <td>0.672816</td>
      <td>0.485259</td>
      <td>0.372151</td>
      <td>0.636666</td>
      <td>0.769526</td>
      <td>0.444131</td>
      <td>0.494401</td>
      <td>0.557343</td>
      <td>0.578228</td>
      <td>0.499506</td>
    </tr>
    <tr>
      <th>irish</th>
      <td>0.562573</td>
      <td>0.905310</td>
      <td>0.499915</td>
      <td>0.317347</td>
      <td>0.472240</td>
      <td>0.781788</td>
      <td>0.488824</td>
      <td>0.434393</td>
      <td>1.000000</td>
      <td>0.552946</td>
      <td>0.553621</td>
      <td>0.383224</td>
      <td>0.281883</td>
      <td>0.476361</td>
      <td>0.418346</td>
      <td>0.851059</td>
      <td>0.863347</td>
      <td>0.527830</td>
      <td>0.345491</td>
      <td>0.322081</td>
    </tr>
    <tr>
      <th>italian</th>
      <td>0.594306</td>
      <td>0.536510</td>
      <td>0.620576</td>
      <td>0.379484</td>
      <td>0.431508</td>
      <td>0.767632</td>
      <td>0.784062</td>
      <td>0.436468</td>
      <td>0.552946</td>
      <td>1.000000</td>
      <td>0.518094</td>
      <td>0.382874</td>
      <td>0.287963</td>
      <td>0.625961</td>
      <td>0.506280</td>
      <td>0.602125</td>
      <td>0.636225</td>
      <td>0.730710</td>
      <td>0.461890</td>
      <td>0.426894</td>
    </tr>
    <tr>
      <th>jamaican</th>
      <td>0.695495</td>
      <td>0.558891</td>
      <td>0.664057</td>
      <td>0.545463</td>
      <td>0.622334</td>
      <td>0.577509</td>
      <td>0.541020</td>
      <td>0.672816</td>
      <td>0.553621</td>
      <td>0.518094</td>
      <td>1.000000</td>
      <td>0.535343</td>
      <td>0.438136</td>
      <td>0.634928</td>
      <td>0.676616</td>
      <td>0.516025</td>
      <td>0.654692</td>
      <td>0.633628</td>
      <td>0.637914</td>
      <td>0.568927</td>
    </tr>
    <tr>
      <th>japanese</th>
      <td>0.460317</td>
      <td>0.395659</td>
      <td>0.473538</td>
      <td>0.891298</td>
      <td>0.790714</td>
      <td>0.408776</td>
      <td>0.354222</td>
      <td>0.485259</td>
      <td>0.383224</td>
      <td>0.382874</td>
      <td>0.535343</td>
      <td>1.000000</td>
      <td>0.867110</td>
      <td>0.428084</td>
      <td>0.351411</td>
      <td>0.424547</td>
      <td>0.464781</td>
      <td>0.446541</td>
      <td>0.694407</td>
      <td>0.757926</td>
    </tr>
    <tr>
      <th>korean</th>
      <td>0.363876</td>
      <td>0.276495</td>
      <td>0.419077</td>
      <td>0.861580</td>
      <td>0.691773</td>
      <td>0.281242</td>
      <td>0.255248</td>
      <td>0.372151</td>
      <td>0.281883</td>
      <td>0.287963</td>
      <td>0.438136</td>
      <td>0.867110</td>
      <td>1.000000</td>
      <td>0.337405</td>
      <td>0.249926</td>
      <td>0.306497</td>
      <td>0.351543</td>
      <td>0.365586</td>
      <td>0.605315</td>
      <td>0.684010</td>
    </tr>
    <tr>
      <th>mexican</th>
      <td>0.676835</td>
      <td>0.451349</td>
      <td>0.630519</td>
      <td>0.450431</td>
      <td>0.504062</td>
      <td>0.541181</td>
      <td>0.580871</td>
      <td>0.636666</td>
      <td>0.476361</td>
      <td>0.625961</td>
      <td>0.634928</td>
      <td>0.428084</td>
      <td>0.337405</td>
      <td>1.000000</td>
      <td>0.589040</td>
      <td>0.510936</td>
      <td>0.627142</td>
      <td>0.643414</td>
      <td>0.574680</td>
      <td>0.537445</td>
    </tr>
    <tr>
      <th>moroccan</th>
      <td>0.509188</td>
      <td>0.410732</td>
      <td>0.493638</td>
      <td>0.335313</td>
      <td>0.359234</td>
      <td>0.522960</td>
      <td>0.642514</td>
      <td>0.769526</td>
      <td>0.418346</td>
      <td>0.506280</td>
      <td>0.676616</td>
      <td>0.351411</td>
      <td>0.249926</td>
      <td>0.589040</td>
      <td>1.000000</td>
      <td>0.458994</td>
      <td>0.470758</td>
      <td>0.643223</td>
      <td>0.446732</td>
      <td>0.412660</td>
    </tr>
    <tr>
      <th>russian</th>
      <td>0.586674</td>
      <td>0.883071</td>
      <td>0.516681</td>
      <td>0.352506</td>
      <td>0.504358</td>
      <td>0.847700</td>
      <td>0.568039</td>
      <td>0.444131</td>
      <td>0.851059</td>
      <td>0.602125</td>
      <td>0.516025</td>
      <td>0.424547</td>
      <td>0.306497</td>
      <td>0.510936</td>
      <td>0.458994</td>
      <td>1.000000</td>
      <td>0.814122</td>
      <td>0.614375</td>
      <td>0.370934</td>
      <td>0.365120</td>
    </tr>
    <tr>
      <th>southern_us</th>
      <td>0.676681</td>
      <td>0.875686</td>
      <td>0.675126</td>
      <td>0.453500</td>
      <td>0.581332</td>
      <td>0.802111</td>
      <td>0.547778</td>
      <td>0.494401</td>
      <td>0.863347</td>
      <td>0.636225</td>
      <td>0.654692</td>
      <td>0.464781</td>
      <td>0.351543</td>
      <td>0.627142</td>
      <td>0.470758</td>
      <td>0.814122</td>
      <td>1.000000</td>
      <td>0.644282</td>
      <td>0.467339</td>
      <td>0.425230</td>
    </tr>
    <tr>
      <th>spanish</th>
      <td>0.727687</td>
      <td>0.503061</td>
      <td>0.774291</td>
      <td>0.461079</td>
      <td>0.512150</td>
      <td>0.754748</td>
      <td>0.726851</td>
      <td>0.557343</td>
      <td>0.527830</td>
      <td>0.730710</td>
      <td>0.633628</td>
      <td>0.446541</td>
      <td>0.365586</td>
      <td>0.643414</td>
      <td>0.643223</td>
      <td>0.614375</td>
      <td>0.644282</td>
      <td>1.000000</td>
      <td>0.564578</td>
      <td>0.503979</td>
    </tr>
    <tr>
      <th>thai</th>
      <td>0.686447</td>
      <td>0.338618</td>
      <td>0.543496</td>
      <td>0.718632</td>
      <td>0.736732</td>
      <td>0.443445</td>
      <td>0.438834</td>
      <td>0.578228</td>
      <td>0.345491</td>
      <td>0.461890</td>
      <td>0.637914</td>
      <td>0.694407</td>
      <td>0.605315</td>
      <td>0.574680</td>
      <td>0.446732</td>
      <td>0.370934</td>
      <td>0.467339</td>
      <td>0.564578</td>
      <td>1.000000</td>
      <td>0.908089</td>
    </tr>
    <tr>
      <th>vietnamese</th>
      <td>0.557734</td>
      <td>0.321151</td>
      <td>0.501517</td>
      <td>0.783786</td>
      <td>0.771097</td>
      <td>0.404030</td>
      <td>0.412951</td>
      <td>0.499506</td>
      <td>0.322081</td>
      <td>0.426894</td>
      <td>0.568927</td>
      <td>0.757926</td>
      <td>0.684010</td>
      <td>0.537445</td>
      <td>0.412660</td>
      <td>0.365120</td>
      <td>0.425230</td>
      <td>0.503979</td>
      <td>0.908089</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the size of the matrix
print("Correlation matrix shape:", cuisine_correlation.shape)
print("Correlation matrix size:", cuisine_correlation.size)
```

    Correlation matrix shape: (20, 20)
    Correlation matrix size: 400
    

## Plot the correlation matrix

Visualise the correlation between cuisines and their shared ingredients


```python
# Visualise the data on a correlation matrix
plt.figure(figsize=(15, 7))
plt.imshow(cuisine_correlation, cmap="YlGnBu", aspect="auto")

# Add numerical correlation values to each cell in the heatmap
for i in range(cuisine_correlation.shape[0]):
    for j in range(cuisine_correlation.shape[1]):
        plt.text(j, i, f"{cuisine_correlation.values[i, j]:.2f}", ha="center", va="center", color="black")

# To finalize the heatmap with proper labels and titles
plt.xticks(np.arange(len(cuisine_correlation.columns)), cuisine_correlation.columns, rotation=45, ha="right")
plt.yticks(np.arange(len(cuisine_correlation.index)), cuisine_correlation.index)
plt.colorbar(label="Correlation")
plt.title("Cuisine Similarity Based on Ingredients (Correlation)")
plt.xlabel("Cuisine")
plt.ylabel("Cuisine")
plt.tight_layout()
plt.show()
```


    
![png](output_37_0.png)
    


## Machine Learning

Partition the dataset with variables Response (Cuisine) and Predictor (Ingredients). As usual, the train size is 75% while the test set is the remaining 25%. The Y datasets are flattened into 1D NumPy array for compatability with the models


```python
X = pd.DataFrame(recipe_data['ingredients_lem'])
y = pd.DataFrame(recipe_data['cuisine'])

# Split into test set (25%) and training set (75%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42,stratify=y)

# Convert to NumPy array and flatten into 1D array
y_train_flat = y_train.values.flatten()
y_test_flat = y_test.values.flatten()

# Check the sample sizes
print("Train Set:", X_train.shape, y_train_flat.shape)
print("Test Set:", X_test.shape, y_test_flat.shape)
```

    Train Set: (29757, 1) (29757,)
    Test Set: (9920, 1) (9920,)
    

### Transform the Datasets

Ingredient arrays are merged into a single string, and their count are recorded into a Bag-of-words (BoW) so the model can identify their frequencies in the cuisines 


```python
# Step 1: Transform train & test sets
# Join ingredients into a single string with comma for each recipe
X_train['ingredients_str'] = X_train['ingredients_lem'].apply(lambda x:  ', '.join(x))
X_test['ingredients_str'] = X_test['ingredients_lem'].apply(lambda x:  ', '.join(x))

# Step 3: Feature Engineering (NLP) into bag of words
# Custom tokenizer function (splits ingredient names separated by commas)
def custom_tokenizer(text):
    # Split the input string by comma, remove extra spaces, and return the list of ingredients
    return [ingredient.strip() for ingredient in text.split(",")]

vectoriser = CountVectorizer(tokenizer = custom_tokenizer)

X_train_bow = vectoriser.fit_transform(X_train['ingredients_str']).toarray()
X_test_bow = vectoriser.transform(X_test['ingredients_str']).toarray()

print(vectoriser.get_feature_names_out()[100:200])

# Get the size of the train set and the vectorizer object
train_size, vector_size = X_train_bow.shape

print("Number of train ingredients:", train_size)
print("Train vector size:", vector_size)
```

    c:\Users\Rayson\anaconda3\Lib\site-packages\sklearn\feature_extraction\text.py:528: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
      warnings.warn(
    

    ['amchur' 'america' 'american cheese' 'american cheese food'
     'american cheese slices' 'amontillado sherry' 'ampalaya' 'anaheim chile'
     'anasazi beans' 'ancho' 'ancho chile pepper' 'ancho chili ground pepper'
     'ancho powder' 'anchovy' 'anchovy filets' 'anchovy fillets'
     'anchovy paste' 'and carrot green pea' 'and cook drain pasta ziti'
     'and fat free half half' 'andouille chicken sausage' 'andouille sausage'
     'andouille sausage links' 'andouille turkey sausages' 'angel food cake'
     'angel food cake mix' 'angel hair' 'angostura bitters' 'anise'
     'anise basil' 'anise extract' 'anise liqueur' 'anise oil' 'anise powder'
     'anise seed' 'anisette' 'anjou pears' 'annatto' 'annatto oil'
     'annatto powder' 'annatto seeds' 'any' 'aonori' 'apple' 'apple brandy'
     'apple butter' 'apple cider' 'apple cider vinegar' 'apple jelly'
     'apple juice' 'apple juice concentrate' 'apple pie filling'
     'apple pie spice' 'apple puree' 'apple schnapps' 'apple slice'
     'applesauce' 'applewood smoked bacon' 'apricot' 'apricot brandy'
     'apricot halves' 'apricot jam' 'apricot nectar' 'apricot preserves'
     'aquavit' 'arame' 'arbol chile' 'arborio rice' 'arctic char'
     'arepa flour' 'argo corn starch' 'arhar' 'arhar dal' 'armagnac'
     'arrow root' 'arrowroot' 'arrowroot flour' 'arrowroot powder'
     'arrowroot starch' 'artichok heart marin' 'artichoke' 'artichoke bottoms'
     'artichoke hearts' 'artificial sweetener' 'artisan bread' 'arugula'
     'asadero' 'asafetida' 'asafetida (powder)' 'asafetida powder'
     'asafoetida' 'asafoetida powder' 'asakusa nori' 'ascorbic acid' 'asiago'
     'asian barbecue sauce' 'asian basil' 'asian black bean sauce'
     'asian chile paste' 'asian chili red sauc']
    Number of train ingredients: 29757
    Train vector size: 6183
    

### Random Forest Model

Create a Random Forest Model and fit the BoW and the flattened Y values. A classification report is printed once the fitting is done. For a train set of size ~30000, a 70-80% accuracy is the expected standard for a good model performance.

The default number of trees in the model is 100. We will shorten it down to 50 in order to reduce runtime though with negligible decrease in accuracy (0.002)


```python
rf_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Fit train data to Random Forest model
rf_model.fit(X_train_bow, y_train_flat)

# Use model to make predictions on test data
y_pred_rf = rf_model.predict(X_test_bow)

# Evaluate model performance
print("Accuracy of Random Forest model:", accuracy_score(y_test_flat, y_pred_rf))
print(classification_report(y_test_flat, y_pred_rf))

```

    Accuracy of Random Forest model: 0.7179435483870967
                  precision    recall  f1-score   support
    
       brazilian       0.73      0.41      0.53       116
         british       0.67      0.29      0.40       201
    cajun_creole       0.78      0.67      0.72       385
         chinese       0.68      0.86      0.76       667
        filipino       0.60      0.41      0.49       189
          french       0.55      0.53      0.54       659
           greek       0.74      0.57      0.64       293
          indian       0.78      0.89      0.83       749
           irish       0.56      0.35      0.43       167
         italian       0.70      0.88      0.78      1958
        jamaican       0.90      0.49      0.64       130
        japanese       0.76      0.59      0.67       354
          korean       0.81      0.55      0.65       207
         mexican       0.81      0.90      0.85      1607
        moroccan       0.84      0.60      0.70       205
         russian       0.56      0.29      0.38       122
     southern_us       0.67      0.72      0.69      1075
         spanish       0.61      0.22      0.32       247
            thai       0.75      0.65      0.70       384
      vietnamese       0.62      0.39      0.48       205
    
        accuracy                           0.72      9920
       macro avg       0.71      0.56      0.61      9920
    weighted avg       0.72      0.72      0.70      9920
    
    

## AUC-ROC Random Forest

Receiver Operating Characteristic (ROC) curve represents the relationship of True Positive Rate (TPR) against False Positive Rate (FPR)

Area Under the ROC Curve summarises the performance of the model based on classification. It ranges between 0 to 1. A value of 0.5 indicates the model is performing no better than random guessing, and below 0.5 means it will be worse. 


```python
# Generate predicted probabilities (for AUC-ROC calculation)
y_pred_proba_rf = rf_model.predict_proba(X_test_bow)
print(y_pred_proba_rf.shape)
print(y_test_flat)

from sklearn.preprocessing import LabelBinarizer

# Binarize y_test_flat to get one-vs-rest format for ROC curve calculation
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test_flat)

# Step 3: Calculate AUC-ROC score (One-vs-Rest strategy for multi-class classification)
try:
    auc_rf = roc_auc_score(y_test_binarized, y_pred_proba_rf, multi_class='ovr')
    print("Area Under Curve (AUC) for Random Forest model ROC curve:", auc_rf)
except ValueError as e:
    print("AUC calculation error:", e)

# Step 4: Plot the ROC Curve for each class
unique_classes = np.unique(y_test_flat)
print("Unique classes in y_test_flat:", unique_classes)
n_classes = len(unique_classes)
print(n_classes)
fpr_rf = {}
tpr_rf = {}
roc_auc_rf = {}

# Calculate ROC for each class
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

```

    (9920, 20)
    ['thai' 'mexican' 'mexican' ... 'mexican' 'mexican' 'chinese']
    Area Under Curve (AUC) for Random Forest model ROC curve: 0.9360337348898227
    Unique classes in y_test_flat: ['brazilian' 'british' 'cajun_creole' 'chinese' 'filipino' 'french'
     'greek' 'indian' 'irish' 'italian' 'jamaican' 'japanese' 'korean'
     'mexican' 'moroccan' 'russian' 'southern_us' 'spanish' 'thai'
     'vietnamese']
    20
    


```python
# Plot all ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    if not np.isnan(roc_auc_rf[i]):  # Only plot if AUC was calculated
        plt.plot(fpr_rf[i], tpr_rf[i], lw=2, label=f'Class {unique_classes[i]} (AUC = {roc_auc_rf[i]:.2f})')

# Plot the random chance line (diagonal)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc='lower right')
plt.show()
```


    
![png](output_46_0.png)
    


### Feature Importance

A huge advantage of Random Forest is its built-in Feature Importance ability that shows the most important features in determining a prediction. Thus, we will show the top 10 key ingredients used to predict the cuisines.


```python
#Check for importance of ingredients for RandomForest model (Feature Importance)

importances = rf_model.feature_importances_
feature_names = vectoriser.get_feature_names_out()


importance_df = pd.DataFrame({'Ingredient': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10,6))
sb.barplot(x="Importance",y="Ingredient", data=importance_df.head(10))
plt.title('Top 10 Important Ingredients for Cuisine Prediction')
plt.show()

```


    
![png](output_48_0.png)
    


## Logistic Regression Model

Create a Logistic Regression model with max_iter = 300, a reasonable value to balance between convergence issues (too low) and computation time (too high)

Similar to Random Forest, a classification report will be output


```python
# Create a multinomial logistic regression model
logreg_model = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter = 300).fit(X_train_bow, y_train_flat)

#Fit train data to logistic regression model
logreg_model.fit(X_train_bow, y_train_flat)

# Use model to make predictions on test data
y_pred_logreg = logreg_model.predict(X_test_bow)

# Evaluate model performance
print("Accuracy of the best Logistic Regression model:", accuracy_score(y_test_flat, y_pred_logreg))
print(classification_report(y_test_flat, y_pred_logreg))
```

    Accuracy of the best Logistic Regression model: 0.7731854838709677
                  precision    recall  f1-score   support
    
       brazilian       0.75      0.49      0.59       116
         british       0.58      0.41      0.48       201
    cajun_creole       0.78      0.69      0.73       385
         chinese       0.78      0.84      0.81       667
        filipino       0.74      0.57      0.64       189
          french       0.58      0.65      0.62       659
           greek       0.79      0.66      0.72       293
          indian       0.87      0.88      0.88       749
           irish       0.58      0.46      0.51       167
         italian       0.78      0.89      0.83      1958
        jamaican       0.84      0.67      0.75       130
        japanese       0.80      0.68      0.74       354
          korean       0.81      0.72      0.76       207
         mexican       0.89      0.92      0.90      1607
        moroccan       0.87      0.76      0.81       205
         russian       0.70      0.41      0.52       122
     southern_us       0.71      0.78      0.74      1075
         spanish       0.58      0.48      0.53       247
            thai       0.79      0.73      0.76       384
      vietnamese       0.67      0.58      0.62       205
    
        accuracy                           0.77      9920
       macro avg       0.75      0.66      0.70      9920
    weighted avg       0.77      0.77      0.77      9920
    
    

## AUC-ROC Logistic Regression

Similar to Random Forest, An AUC-ROC evaluation will be run for Logistic Regression


```python
# Generate predicted probabilities (for AUC-ROC calculation)
y_pred_proba_lr = logreg_model.predict_proba(X_test_bow)
print(y_pred_proba_lr.shape)
print(y_test_flat)

from sklearn.preprocessing import LabelBinarizer

# Binarize y_test_flat to get one-vs-rest format for ROC curve calculation
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test_flat)

# Step 3: Calculate AUC-ROC score (One-vs-Rest strategy for multi-class classification)
try:
    auc_lr = roc_auc_score(y_test_binarized, y_pred_proba_lr, multi_class='ovr')
    print("Area Under Curve (AUC) for Log Reg model ROC curve:", auc_lr)
except ValueError as e:
    print("AUC calculation error:", e)

# Step 4: Plot the ROC Curve for each class
print("Unique classes in y_test_flat:", np.unique(y_test_flat))
unique_classes = np.unique(y_test_flat)
n_classes = len(unique_classes)
print(n_classes)
fpr_lr = {}
tpr_lr = {}
roc_auc_lr = {}

# Calculate ROC for each class
for i in range(n_classes):
    fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_lr[:, i])
    roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

```

    (9920, 20)
    ['thai' 'mexican' 'mexican' ... 'mexican' 'mexican' 'chinese']
    Area Under Curve (AUC) for Log Reg model ROC curve: 0.9687128227306534
    Unique classes in y_test_flat: ['brazilian' 'british' 'cajun_creole' 'chinese' 'filipino' 'french'
     'greek' 'indian' 'irish' 'italian' 'jamaican' 'japanese' 'korean'
     'mexican' 'moroccan' 'russian' 'southern_us' 'spanish' 'thai'
     'vietnamese']
    20
    


```python
# Plot all ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    if not np.isnan(roc_auc_lr[i]):  # Only plot if AUC was calculated
        plt.plot(fpr_lr[i], tpr_lr[i], lw=2, label=f'Class {unique_classes[i]} (AUC = {roc_auc_lr[i]:.2f})')

# Plot the random chance line (diagonal)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Logistic Regression')
plt.legend(loc='lower right')
plt.show()
```


    
![png](output_53_0.png)
    


## Feature Importance

Unlike Random Forest, Logistic Regression does not have a built-in feature importance function. So we will instead manually calculate the importance by extracting the model coefficients, finding their average using the numpy.mean() function and extracting their features with the count_vectoriser.


```python
# Check for importance of ingredients for Log Reg model (Feature Importance)
coefficients = logreg_model.coef_


# Calculate feature importance by taking the average of the absolute values of the coefficients across all classes.
avg_importance = np.mean(np.abs(coefficients), axis = 0)
feature_names = vectoriser.get_feature_names_out()

feature_importance = pd.DataFrame({"Ingredients": feature_names, "Importance": avg_importance})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
#feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

plt.figure(figsize=(10,6))
sb.barplot(x = "Importance", y = "Ingredients", data= feature_importance.head(10))
plt.title('Top 10 Important Ingredients for Cuisine Prediction')
plt.show()
```


    
![png](output_55_0.png)
    


## Confusion Matrix

Visual representation of the percentage of correct classifications and misclassifications, across the different cuisines for both models


```python
# Model Evaluation
# Plot the Confusion Matrix for both models
cuisine_names = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french',
                 'greek', 'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean',
                  'moroccan', 'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

f, axes = plt.subplots(2, 1, figsize=(16, 32))

# Confusion matrix for Random Forest model
sb.heatmap(confusion_matrix(y_test_flat, y_pred_rf, normalize='true'), # Normalize to percentages
           annot=True, fmt=".1f", annot_kws={"size": 20}, ax=axes[0], cmap="magma",
           xticklabels=cuisine_names, yticklabels=cuisine_names)
axes[0].set_title("Random Forest Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Confusion matrix for Logistic Regression model
sb.heatmap(confusion_matrix(y_test_flat, y_pred_logreg, normalize='true'),  # Normalize to percentages
           annot=True, fmt=".1f", annot_kws={"size": 20}, ax=axes[1], cmap="magma",
           xticklabels=cuisine_names, yticklabels=cuisine_names)
axes[1].set_title("Logistic Regression Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.show()
```


    
![png](output_57_0.png)
    

ploading SoCooked.md…]()
