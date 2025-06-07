# The Satisfaction of Food; The Downfall of the World: A Machine Learning project that investigates Ratings on Recipes
A Data Science project that investigates a recipes dataset to create a machine learning model to predict ratings of recipes

By Yash Tandon and Shiv Mehta

## Intro

### Datascience Question

Is there a relationship between recipe ratings and the given features of the dataset and how reliably can we predict it?

### Our Dataset 

The dataset we have chosen for this project is the recipe rating dataset from food.com. This dataset encompasses a large amount of recipes each with their own subset of ratings and reviews along with additional data regarding the ingredients, listed tags, date of submission and more! In this project we will focus on analyzing the trends and patterns between the features for each recipe and their corresponding ratings to derive insights regarding the same. The reason we chose it because we have become very health conscious in recent years. We go to the gym together and we make a lot of meals in our apartment. Concerned about the health of the United States, we wanted to do a study to see how our country views recipes, especially those which are healthier than others. Through our study, one can find out whether it's possible to describe how well a recipe is liked through the features that it composes of, giving us insight into what makes recipes popular. 

### Describing the Dataset
We are working with two seperate datasets. The first one consists of the Recipe information, and has 83782 rows representing unique recipes. 

| Columm Name | Description |
|----------|----------|
| `name`      | Name of the recipe       |
| `id`       | Unique ID given to the recipe       |
| `minutes`       | Number of minutes to make the recipe       |
| `contributor_id`       | Unique ID of the user who uploaded the recipe       |
| `submitted`       | The date of recipe submission       |
| `tags`       | Tags attributed to the recipe       |
| `nutrition`       | Nutrition values in Percentage of Daily Value (PDV) of each serving of the recipe in the following format: [Number of calories, total fat in PDV, sugar in PDV, sodium in PDV, protein in PDV, saturated fat in PDV, carbohydrates in PDV]       |
| `n_steps`       | Number of steps to make the recipe       |
| `steps`       | The written description of the steps to make the recipe       |
| `description`      | The text submitted by the user to describe the recipe      |
| `ingredients`      | The list of ingredients required to make the recipe      |
| `n_ingredients`      | The number of ingredients required to make the recipe      |

The other dataset consists of ratings and reviews for those recipes, and it has 731927 rows representing unique reviews left by users.

| Columm Name | Description |
|----------|----------|
| `user_id`      | Unique ID of the user who reviewed the recipe       |
| `recipe_id`       | Unique ID given to the recipe       |
| `date`       | The date of review submission       |
| `rating`       | A integer rating, from 0 to 5, given to the recipe by the user       |
| `review`       | A textual description of the user's review of the recipe       |

## EDA and Data Cleaning
### Steps of Cleaning

In order to clean and tidy the dataset for analysis, we took the following steps:

1. Merging
	- To start, we merged the recipes dataframe with the reviews dataframe. we use left merge here to ensure that all of the recipes, even those without ratings, are kept. 
2. Renaming and Typecasting Columns
	- Next, we renamed the columns to have appropriate feature names, especially after merging. Names like "date" and "submission" are especially confusing without context.
	- Following that we typecasted all relevant columns to have appropriate typing. 
	- First we had nominal variables stored as ints so we convert them to strings since we'll never use math on them and prevent formatting issues. 
	- We also converted dates to pandas date-time objects to make applying functions simpler. 
	- Lastly we wanted to convert lists that are stored as strings of a list of strings to a list of just the strings for easier data manipulation and faster access
3. Creating new Columns for Relevant Features
	- Next we created new feature columns out of existing ones for better analysis
	- To start we separated the 'nutrition' column into individual components for 'calories', 'total_fat_PDV', 'sugar_PDV', 'sodium_PDV', 'protein_PDV', 'saturated_fat_PDV', 'carbohydrates_PDV' respectively. Previously accessing through functions would have required hardcoding list indices or a dictionary for mapping, now access is much easier and so are computations.
	- Speaking of computations; We converted protein PDV to amount of protein in calories with respect to total calories of the recipe in order to add a feature which was informative with respect to the whole calories in the recipe to analyze health and to add a multiplicative feature which the model would not explore on its own. We did this by converting the protein percentage into its value in grams and converting that to calories since each gram of protein has 4 calories, finally dividing by the total number of calories to get the protein-calorie ratio. 
4. Adding Average and Bayesian Ratings
	- The last features we added were average and bayesian ratings. 
		- Average ratings are given by the mean of each recipe
		- Bayesian ratings give a more informed average rating for each recipe as it accounts for not only the recipe's true mean but also the skew of ratings in the dataset by incorporating the overall recipe mean. 
			- The Bayesian average is computed as: \[\text{Bayesian Rating} = \frac{v}{v + m} \cdot R + \frac{m}{v + m} \cdot C\], where:
            
            \( R \) = average rating for the recipe
            
            \( v \) = number of ratings for the recipe
            
            \( m \) = minimum number of ratings required to be considered (smoothing constant) which is chosen by us. We chose 4 because it's distribution was the closest to a normal distribution after various trials
            
            \( C \) = global average rating across all recipes


5. Fixing NaN Nalues
Lastly we had to fix all the NaN values of our final cleaned dataset
	- Ratings are usually on 1-5 scale. We know that ratings of 0 are representative of missing ratings. This means that all recipes with calories=0 have NaNs due to error from DivisionByZero so we can replace them with 0

### Columns of cleaned and tidied DataFrame

| Column Name            | Data Type      |
|:-----------------------|:---------------|
| recipe_name            | object         |
| recipe_id              | object         |
| minutes                | int64          |
| contributor_id         | object         |
| recipe_submission_date | datetime64[ns] |
| tags                   | object         |
| n_steps                | int64          |
| steps                  | object         |
| description            | object         |
| ingredients            | object         |
| n_ingredients          | int64          |
| rater_id               | object         |
| review_posted_date     | datetime64[ns] |
| rating                 | float64        |
| review                 | object         |
| calories               | float64        |
| total_fat_PDV          | float64        |
| sugar_PDV              | float64        |
| sodium_PDV             | float64        |
| protein_PDV            | float64        |
| saturated_fat_PDV      | float64        |
| carbohydrates_PDV      | float64        |
| pro_cal_ratio          | float64        |
| average_rating         | float64        |
| bayesian_rating        | float64        |

### Head of Cleaned DataFrame
The cleaned and tidied DataFrame after merging has 234429 rows and 25 columns. The first five rows are displayed below. 

| recipe_name                          |   recipe_id |   minutes |   contributor_id | recipe_submission_date   | tags                                                                                                                                                                                                                        |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                    |   n_ingredients |         rater_id | review_posted_date   |   rating | review                                                                                                                                                                                                                                                                                                                                           |   calories |   total_fat_PDV |   sugar_PDV |   sodium_PDV |   protein_PDV |   saturated_fat_PDV |   carbohydrates_PDV |   pro_cal_ratio |   average_rating |   bayesian_rating |
|:-------------------------------------|------------:|----------:|-----------------:|:-------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------------:|:---------------------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|----------------:|------------:|-------------:|--------------:|--------------------:|--------------------:|----------------:|-----------------:|------------------:|
| 1 brownies in the world    best ever |      333281 |        40 |           985201 | 2008-10-27 00:00:00      | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat ', 'stirring frequently ', 'until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs ', 'sugar ', 'cocoa powder ', 'vanilla extract ', 'espresso ', 'and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean ', 'about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour'] |               9 | 386585           | 2008-11-19 00:00:00  |        4 | These were pretty good, but took forever to bake.  I would send it ended up being almost an hour!  Even then, the brownies stuck to the foil, and were on the overly moist side and not easy to cut.  They did taste quite rich, though!  Made for My 3 Chefs.                                                                                   |      138.4 |              10 |          50 |            3 |             3 |                  19 |                   6 |       0.0433526 |                4 |           4.30378 |
| 1 in canada chocolate chip cookies   |      453467 |        45 |          1848091 | 2011-04-11 00:00:00      | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl ', 'sift together the flours and baking powder', 'set aside', 'in another mixing bowl ', 'blend together the sugars ', 'margarine ', 'and salt until light and fluffy', 'add the eggs ', 'water ', 'and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop ', 'scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !'] | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                    |              11 | 424680           | 2012-01-26 00:00:00  |        5 | Originally I was gonna cut the recipe in half (just the 2 of us here), but then we had a park-wide yard sale, & I made the whole batch & used them as enticements for potential buyers ~ what the hey, a free cookie as delicious as these are, definitely works its magic! Will be making these again, for sure! Thanks for posting the recipe! |      595.1 |              46 |         211 |           22 |            13 |                  51 |                  26 |       0.0436901 |                5 |           4.50378 |
| 412 broccoli casserole               |      306168 |        40 |            50969 | 2008-05-30 00:00:00      | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray ', 'set aside', 'in a large bowl mix together broccoli ', 'soup ', 'one cup of cheese ', 'garlic powder ', 'pepper ', 'salt ', 'milk ', '1 cup of french onions ', 'and soy sauce', 'pour into baking dish ', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly ', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                      | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |  29782           | 2008-12-31 00:00:00  |        5 | This was one of the best broccoli casseroles that I have ever made.  I made my own chicken soup for this recipe. I was a bit worried about the tsp of soy sauce but it gave the casserole the best flavor. YUM!                                                                                                                                  |      194.8 |              20 |           6 |           32 |            22 |                  36 |                   3 |       0.225873  |                5 |           4.68986 |
|                                      |             |           |                  |                          |                                                                                                                                                                                                                             |           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                |                 |                  |                      |          | The photos you took (shapeweaver) inspired me to make this recipe and it actually does look just like them when it comes out of the oven.                                                                                                                                                                                                        |            |                 |             |              |               |                     |                     |                 |                  |                   |
|                                      |             |           |                  |                          |                                                                                                                                                                                                                             |           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                |                 |                  |                      |          | Thanks so much for sharing your recipe shapeweaver. It was wonderful!  Going into my family's favorite Zaar cookbook :)                                                                                                                                                                                                                          |            |                 |             |              |               |                     |                     |                 |                  |                   |
| 412 broccoli casserole               |      306168 |        40 |            50969 | 2008-05-30 00:00:00      | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray ', 'set aside', 'in a large bowl mix together broccoli ', 'soup ', 'one cup of cheese ', 'garlic powder ', 'pepper ', 'salt ', 'milk ', '1 cup of french onions ', 'and soy sauce', 'pour into baking dish ', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly ', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                      | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |      1.19628e+06 | 2009-04-13 00:00:00  |        5 | I made this for my son's first birthday party this weekend. Our guests INHALED it! Everyone kept saying how delicious it was. I was I could have gotten to try it.                                                                                                                                                                               |      194.8 |              20 |           6 |           32 |            22 |                  36 |                   3 |       0.225873  |                5 |           4.68986 |
| 412 broccoli casserole               |      306168 |        40 |            50969 | 2008-05-30 00:00:00      | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray ', 'set aside', 'in a large bowl mix together broccoli ', 'soup ', 'one cup of cheese ', 'garlic powder ', 'pepper ', 'salt ', 'milk ', '1 cup of french onions ', 'and soy sauce', 'pour into baking dish ', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly ', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                      | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 | 768828           | 2013-08-02 00:00:00  |        5 | Loved this.  Be sure to completely thaw the broccoli.  I didn&#039;t and it didn&#039;t get done in time specified.  Just cooked it a little longer though and it was perfect.  Thanks Chef.                                                                                                                                                     |      194.8 |              20 |           6 |           32 |            22 |                  36 |                   3 |       0.225873  |                5 |           4.68986 |

### Univariate Plot: Distribution of Protein-Calorie Ratios across Recipes

We decided to plot the distribution of Protein-Calorie Ratios across all of the recipes in the dataset. As shown below, the distribution is heavily skewed to the right, as the modal Protein-Calorie Ratio is around 0.05. This means that most recipes' calories aren't made up by Protein. Only a very low proportion of recipes have a Protein-Calorie ratio of greater than 0.5.

<iframe
  src="assets/univariate.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
  
### Bivariate Plot: Average Protein-Calorie Ratio by Vegan Status

We plotted a bar chart which displays the average Protein-Calorie ratio of recipes that are and aren't Vegan. A recipe is classified "Vegan" if it contains "Vegan" in its tag list. The recipes that aren't Vegan have a greater average Protein-Calorie Ratio (~0.16) than recipes that are Vegan (~0.10)

<iframe
  src="assets/bivariate.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
  
### Interesting Aggregates

For the Aggregation table, we chose to look at the relationship between the number of ingredients and the nutrition values (saturated fats, protein, carbohydrates, and calories). This information is significant because it can show whether complexity of a recipe can determine it's nutritional content. When choosing features for our model, this information will help us pick variables that are not redundant to avoid multicollinearity. From the table, it's unlikely that this is the case.

|   ('saturated_fat_PDV', 'mean') |   ('saturated_fat_PDV', 'median') |   ('saturated_fat_PDV', 'max') |   ('saturated_fat_PDV', 'min') |   ('saturated_fat_PDV', 'std') |   ('protein_PDV', 'mean') |   ('protein_PDV', 'median') |   ('protein_PDV', 'max') |   ('protein_PDV', 'min') |   ('protein_PDV', 'std') |   ('carbohydrates_PDV', 'mean') |   ('carbohydrates_PDV', 'median') |   ('carbohydrates_PDV', 'max') |   ('carbohydrates_PDV', 'min') |   ('carbohydrates_PDV', 'std') |   ('calories', 'mean') |   ('calories', 'median') |   ('calories', 'max') |   ('calories', 'min') |   ('calories', 'std') |
|--------------------------------:|----------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|----------------------------:|-------------------------:|-------------------------:|-------------------------:|--------------------------------:|----------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|----------------------:|
|                        162.448  |                              33   |                            510 |                              0 |                       207.329  |                   53.0938 |                        17   |                      157 |                        0 |                  60.679  |                         5.71875 |                                 1 |                             66 |                              0 |                        9.20035 |               1151.86  |                   230    |                3590.2 |                   7.8 |              1333.42  |
|                         36.1524 |                               6   |                            947 |                              0 |                        85.0137 |                   29.2898 |                         7   |                     1051 |                        0 |                  60.1483 |                        11.5295  |                                 5 |                            241 |                              0 |                       22.8266  |                395.96  |                   168.85 |               12135.5 |                   0   |               741.313 |
|                         25.3339 |                               7   |                           4610 |                              0 |                        97.2076 |                   17.4635 |                         5   |                     1043 |                        0 |                  57.1687 |                         9.78413 |                                 5 |                            641 |                              0 |                       17.5958  |                283.122 |                   168.1  |               13101.5 |                   0   |               482.809 |
|                         28.9208 |                              10   |                           6875 |                              0 |                        84.1935 |                   18.0672 |                         7   |                     4356 |                        0 |                  50.7197 |                        11.0621  |                                 6 |                           3007 |                              0 |                       34.3639  |                305.826 |                   191.8  |               45609   |                   0   |               646.764 |
|                         32.3791 |                              16   |                           1583 |                              0 |                        61.563  |                   23.5252 |                        10   |                     2030 |                        0 |                  40.9518 |                        10.8479  |                                 6 |                           1440 |                              0 |                       26.2001  |                331.26  |                   231.1  |               17551.6 |                   0   |               485.68  |
|                         33.6145 |                              17   |                           1760 |                              0 |                        61.3411 |                   26.1319 |                        12   |                     2929 |                        0 |                  41.9382 |                        11.9193  |                                 7 |                           1511 |                              0 |                       27.4247  |                357.845 |                   248.65 |               17554   |                   0   |               507.021 |
|                         38.5601 |                              20   |                           1528 |                              0 |                        78.7649 |                   28.9474 |                        14   |                     2090 |                        0 |                  40.5359 |                        12.6515  |                                 7 |                            421 |                              0 |                       25.3001  |                393.29  |                   266.1  |                8300.5 |                   0.7 |               572.855 |
|                         36.4596 |                              22   |                           1375 |                              0 |                        50.8961 |                   30.1846 |                        16   |                     1023 |                        0 |                  37.7656 |                        11.9433  |                                 8 |                            527 |                              0 |                       18.1041  |                383.002 |                   287.1  |                9702.6 |                   1   |               403.544 |
|                         37.5433 |                              22   |                           2156 |                              0 |                        58.9138 |                   33.4308 |                        20   |                     1871 |                        0 |                  38.6055 |                        13.3507  |                                 9 |                            623 |                              0 |                       22.4559  |                416.447 |                   305.95 |               17280.4 |                   0.3 |               487.267 |
|                         40.6569 |                              24   |                           1984 |                              0 |                        65.7536 |                   35.3496 |                        21   |                      834 |                        0 |                  40.5264 |                        13.5675  |                                 9 |                            683 |                              0 |                       24.5569  |                435.213 |                   318.9  |               18268.7 |                   0.7 |               585.632 |
|                         44.4075 |                              25   |                           2073 |                              0 |                        73.2639 |                   39.3014 |                        26   |                     1361 |                        0 |                  45.7071 |                        14.4767  |                                 9 |                            601 |                              0 |                       26.4273  |                468.779 |                   341.9  |               18656   |                   2.8 |               617.761 |
|                         43.2503 |                              25   |                           2731 |                              0 |                        80.7147 |                   38.7437 |                        25   |                     3605 |                        0 |                  76.8481 |                        14.3794  |                                10 |                            591 |                              0 |                       23.6282  |                464.954 |                   339    |               21497.8 |                   0.3 |               662.517 |
|                         43.8332 |                              28   |                           2082 |                              0 |                        66.5802 |                   40.8885 |                        31   |                     2637 |                        0 |                  47.0287 |                        15.1702  |                                11 |                            704 |                              0 |                       25.853   |                484.344 |                   372    |               15309.6 |                   1.5 |               578     |
|                         49.375  |                              29   |                           1848 |                              0 |                        73.1217 |                   43.1115 |                        34   |                      443 |                        0 |                  38.2187 |                        15.5417  |                                11 |                            560 |                              0 |                       23.5781  |                514.04  |                   396.6  |               11036.6 |                   2   |               559.656 |
|                         50.2975 |                              31   |                           1837 |                              0 |                        73.5116 |                   48.363  |                        40   |                      461 |                        0 |                  40.7664 |                        16.8685  |                                11 |                            471 |                              0 |                       28.9226  |                552.521 |                   428.7  |               22371.2 |                   4.8 |               680.742 |
|                         59.2786 |                              35   |                           6269 |                              0 |                       219.159  |                   54.8272 |                        48   |                     2246 |                        0 |                  69.7394 |                        18.0631  |                                12 |                            985 |                              0 |                       44.0758  |                609.078 |                   453.7  |               28930.2 |                  19.2 |              1184.07  |
|                         53.4712 |                              38   |                           1527 |                              0 |                        68.828  |                   49.5428 |                        41   |                      505 |                        0 |                  42.283  |                        16.1874  |                                13 |                            616 |                              0 |                       21.8293  |                554.704 |                   446.9  |               13029.1 |                   2.5 |               533.391 |
|                         60.3711 |                              35   |                            713 |                              0 |                        71.5394 |                   53.2127 |                        44   |                      444 |                        0 |                  41.2329 |                        18.0922  |                                14 |                            279 |                              0 |                       24.1081  |                617.204 |                   477.5  |                5329   |                  12.3 |               568.928 |
|                         50.037  |                              32   |                           1986 |                              0 |                        88.8504 |                   50.315  |                        47.5 |                      234 |                        0 |                  35.4362 |                        18.509   |                                13 |                            439 |                              1 |                       34.2697  |                585.077 |                   461.65 |                9131.8 |                  40.2 |               723.439 |
|                         57.245  |                              40   |                            649 |                              0 |                        59.2874 |                   62.9886 |                        53   |                      798 |                        0 |                  49.912  |                        18.5119  |                                15 |                            191 |                              1 |                       19.9175  |                662.043 |                   505.9  |                6551.1 |                  24.4 |               541.858 |
|                         72.167  |                              53   |                            658 |                              0 |                        84.866  |                   66.8774 |                        53   |                      451 |                        1 |                  72.8566 |                        22.1155  |                                14 |                            293 |                              0 |                       30.7078  |                778.759 |                   543.9  |                6827.9 |                  60.7 |               783.509 |
|                         81.0695 |                              70   |                           1146 |                              0 |                        74.3009 |                   73.6042 |                        70   |                     2281 |                        2 |                 141.973  |                        15.937   |                                11 |                            218 |                              1 |                       17.2734  |                701.687 |                   773.5  |               10952.1 |                  38.6 |               738.594 |
|                         61.0351 |                              32   |                            809 |                              0 |                        93.5801 |                   48.5911 |                        37   |                      415 |                        3 |                  40.8955 |                        19.4665  |                                13 |                            429 |                              1 |                       42.0892  |                660.945 |                   511.3  |                9456.2 |                 133.2 |               932.528 |
|                         48.4034 |                              33   |                            366 |                              0 |                        54.997  |                   59.5852 |                        44   |                      344 |                        5 |                  44.8643 |                        16.9148  |                                14 |                            165 |                              2 |                       15.215   |                591.681 |                   454.5  |                3753.1 |                 135.7 |               426.11  |
|                         77.75   |                              51.5 |                            284 |                              1 |                        76.3952 |                   68.4333 |                        51   |                      263 |                        5 |                  56.4066 |                        20.6833  |                                16 |                             90 |                              0 |                       16.3152  |                766.61  |                   703.25 |                1885.9 |                  79.6 |               446.444 |
|                         73.15   |                              57   |                            442 |                              6 |                        86.5453 |                   91.27   |                        81   |                      278 |                        7 |                  58.1803 |                        27.65    |                                23 |                             94 |                              1 |                       18.6279  |                840.441 |                   823.5  |                3267.5 |                 169.6 |               479.949 |
|                        126.356  |                              73   |                           2319 |                              0 |                       339.524  |                   70.3778 |                        79   |                      446 |                        4 |                  72.6881 |                        57.4889  |                                18 |                           1554 |                              5 |                      228.992   |               1301.2   |                   722.6  |               26604.4 |                  77.2 |              3883.04  |
|                         47.8605 |                              36   |                            113 |                              4 |                        31.1613 |                   74.8372 |                        62   |                      146 |                       10 |                  34.4466 |                        24.3256  |                                20 |                            107 |                              5 |                       28.2461  |                670.14  |                   544.8  |                1519.7 |                 139.9 |               377.66  |
|                         44.7097 |                               9   |                            354 |                              5 |                        75.5898 |                   96.3871 |                        60   |                      652 |                       18 |                 115.363  |                        34.6129  |                                31 |                             81 |                              8 |                       21.3005  |                880.897 |                   607.4  |                3756.6 |                 274.1 |               742.738 |
|                         70.2424 |                              68   |                            176 |                              6 |                        46.7186 |                   56.0606 |                        47   |                      123 |                       16 |                  34.301  |                        19.9697  |                                18 |                             38 |                              6 |                        7.99408 |                678.464 |                   651.9  |                1390.2 |                 248.6 |               310.089 |
|                         58.2308 |                              38   |                            134 |                              3 |                        52.4439 |                   39      |                        36   |                      123 |                       13 |                  30.1496 |                        25.0769  |                                11 |                             92 |                              8 |                       30.3438  |                872.454 |                   559    |                1760.2 |                 203.4 |               637.072 |
|                        100.75   |                             114   |                            114 |                             61 |                        26.5    |                   80      |                        94   |                       94 |                       38 |                  28      |                        25.75    |                                33 |                             33 |                              4 |                       14.5     |                864.475 |                  1031.6  |                1031.6 |                 363.1 |               334.25  |
|                         12      |                              12   |                             12 |                             12 |                       nan      |                    8      |                         8   |                        8 |                        8 |                 nan      |                        14       |                                14 |                             14 |                             14 |                      nan       |                338.2   |                   338.2  |                 338.2 |                 338.2 |               nan     |
|                        802      |                             802   |                            802 |                            802 |                       nan      |                   59      |                        59   |                       59 |                       59 |                 nan      |                        26       |                                26 |                             26 |                             26 |                      nan       |              10687.7   |                 10687.7  |               10687.7 |               10687.7 |               nan     |

## Missingness Mechanisms
In our merged dataset, there are 7 columns that have atleast one NaN value. These are `recipe_name`, `description`, `review_posted_date`, `rating`, `review`, `average_rating`, `bayesian_rating`. `recipe_name` and `review_posted_date` only have 1 row-value missing. Because there's 100,000+ rows in the merged, cleaned dataset, and also these columns won't be used in the predictive model, these can be ignored and are Missing completely at random.

### NMAR Analysis
`description` and `review` is likely to be NMAR. 

Firstly, `description` is not missing at random because recipes that are very simple, or very popular, don't need a description because the user who posted the recipe is unable to give more context about the recipe. Recipes that are more complicated or less well known may prompt the user submitting it to put a description to help give readers context.

In a similar fashion, `review` is likely to be not missing at random because if people don't have strong opinions or comments about the dish, then they are less likely to include a textual review with their rating. The users that included ratings had some aspect that they could talk about, suggesting strong feelings about the recipe. 


### Missingness Dependency

We believe that `rating` is possibly missing at random as it depends on another column. 

We will first check to see whether the missingness of `rating` is dependent on the `minutes` column.

**Null Hypothesis**: The missingness of the rating column does not depend on the minutes of the recipe

**Alternate Hypothesis**: The missingness of the rating column does depend on the minutes of the recipe

**Test Statistic**: The absolute difference in the mean minutes between recipes that have missing ratings and non-NaN ratings

**Significance Level**: p < 0.01

To test this, a permutation test was run by shuffling the `is_nan_rating` column 1,000 times and calculating the difference in the mean minutes of the recipes with non-NaN and NaN ratings each time.

<iframe
  src="assets/minutes.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed statistic of this test is around 51.45. The resulting p value is 0.107. Since this is greater than the significance level we chose (0.107 > 0.01), we fail to reject the null hypothesis. Thus, it is likely that the missingness of the `rating` column does not depend on the cooking time, measured in minutes, of the recipe.


Next, we will find out whether the missingness of the ratings column depends on the number of steps, `n_steps`.

**Null Hypothesis**: The missingness of the rating column does not depend on the number of steps of the recipe

**Alternate Hypothesis**: The missingness of the rating column does depend on the number of steps of the recipe

**Test Statistic**: The absolute difference in the mean number of steps between recipes that have missing ratings and non-NaN ratings

**Significance Level**: p < 0.01

To test this, a permutation test was run by shuffling the `is_nan_rating` column 1,000 times and calculating the difference in the mean steps of the recipes with non-NaN and NaN ratings each time.

<iframe
  src="assets/steps.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed statistic of this test is around 1.34. The resulting p value is 0.0. Since this is less than the significance level we chose (0.0 < 0.01), we reject the null hypothesis. Thus, it is likely that the missingness of the `rating` column does depend on the number of steps, `n_steps`, of the recipe.

## Hypothesis Testing

The question we want to answer is: Is there a significant difference in the Protein-Calorie ratio between recipes with high average bayesian ratings (>=4.75) and low average bayesian ratings (<4.75)?

Below are the Null and Alternate hypotheses we propose for this test:

\[H_0: \text{There is no difference in the mean Protein-Calorie ratio of highly-rated recipes and lowly-rated recipes}\]

\[H_1: \text{There is a difference in the mean Protein-Calorie ratio of highly-rated recipes and lowly-rated recipes}\]

We chose a significance level of:

\[ {\alpha} = 0.01  \]
because we wanted a statistically significant result if rejecting the null hypothesis to avoid false positive results and build robustness. 

Our test statistic, T, is:

\[T = |P_\text{high-rated} - P_\text{low-rated}|\]
Which comes out to be: \[0.011521277908209948\]
We chose this test statistic because we were testing for a difference not a directional bias (two tailed test). Furthermore it well quantifies the difference we are trying to measure between the two groups and is sensitive to large differences in groups. 

Running a permutation test for 1000 shuffles yielded the following p-value:

\[p = 0.0\]

<iframe
  src="assets/pro_cal.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed statistic of this test is around 0.012. The resulting p value is 0.0. Since this is less than the significance level we chose (0.0 < 0.01), we reject the null hypothesis. Thus, it is likely that there is a difference in the mean Protein-Calorie ratio of highly-rated recipes and lowly-rated recipes.

## Prediction Problem

### The prediction problem we chose to tackle is: What features are the most deterministic when reliably predicting the bayesian rating of a recipe?

Given that bayesian ratings are a continuous variable, this problem is a regression problem. 

Our responsible variable is: Bayesian Ratings (a more data aware and bias adjusted version of the mean as explained above) 

Our metric of choice is: \[R^2\]
This is for the following reasons:
1. Our data is heavily skewed with median being 4.50 on scale of 1 to 5. This means that metrics such as RMSE will reward underfitting models which solely predict the mean and punish those which are smarter but accumulate error due to the large cluster of data centered on the upper end of the scale. 
2. \[R^2\] punishes the models which underfit and rely on the mean, in fact by definition this metric scores models which guess the mean a 0. 
3. A model that captures the obscure variance and trends of the data is rewarded heavily, which is exactly what we are looking for. 

### Data

We split our data into 80-20 train-test split. In order to account for randomness in the split, the same train-test split will be used for both models. 

## Baseline Model
### Baseline Features

Our standard model is a standard bi-variate linear regression model. It uses 2 features:
1. pro_cal_ratio: this is the protein calorie ratio for each recipe and is a quantitative continuous variable
2. n_ingredients: this is the number of ingredients in each recipe and is a quantitative discrete variable 

### Transformations

Our first transformation was transforming the average ratings to Bayesian Ratings (using the formula mentioned above), which will serve as the response variable.

We also transformed the Protein PDV to Protein-Calorie Ratio to make it more intuitive for the reader (informative with respect to the whole calories in the recipe). More importantly, it creates a feature that the model won't explore on it's own because it's multiplicative (quadratic feature).

We used the standardscaler() to normalize our data in order to compare the weights of the features

### Results

Our baseline model had a \[R^2\] of 0.0002. Considerably poor. 

<iframe
  src="assets/baseline.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This is a plot comparing actual bayesian ratings (x-axis) and our baseline model's predicted ratings (y-axis). From this plot the baseline model consistently predicts the same value at all labels thus showing habit of underfitting and instead just predicting the mean.

<iframe
  src="assets/baseline_res1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot compares the residuals of our baseline model with the actual bayesian ratings of our dataset. The residuals form a perfect straight line with little to no variance, meaning the model is most likely predicting the same value repeatedly. These residuals further reinforce the underfitting and mean predicting nature of the model. 

<iframe
  src="assets/baseline_res2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot compares the residuals of our baseline model with the predicted bayesian ratings of our model. The residuals show little to no variance once again. Although the residuals may appear constant in vertical distance, this is actually explained by underfitting and repeated prediction of the mean. 

All in all, this is a poor model to predict Bayesian ratings with the features. This is because the \[R^2\] value is 0.0002. Furthermore, the model shows clear signs of underfitting because it repeatedly predicts the mean because of the large skew in the data.

## Final Model

### Model
For our final model we decided to use a random forest regression model. We chose this model because it averages out multiple decision trees, helping reduce overfitting and under fitting by averaging multiple predictions; reducing variance and improving robust fitting. 

Although one concern we had which came true was the much larger training and inference time. 

### Hyper Parameters

The hyper parameters for this model are:

 1. 'n_estimator': This is the number of decision trees in the forest; more trees improve performance but increase training time. Choices are [100, 200]
 2. 'max_depth': This is the max depth of each tree; deeper models can model more complex trends but have the risk of overfitting. Choices are [10,20]
 3. 'min_samples_leaf': This is the minimum number of samples for a leaf node; higher values can reduce overfit by preventing trees from creating leaves with minimal samples and lower values can capture more detail but have the risk of overfitting. Choices are [5,10]
 4. 'max_features': This is the number of features taken into consideration when each node splits. Choices are ['sqrt', 'log2'] (squareroot and log2)

### Quantitative Features

For this model we chose the features :

1. 'pro_cal_ratio': this is the protein calorie ratio for each recipe and is a quantitative continuous variable. We inferred that there is a statistically significant difference in the protein-calorie ratio between recipes with high and low bayesian ratings, thus making it a relevant feature to consider. 
2. 'n_ingredients': this is the number of ingredients in each recipe and is a quantitative discrete variable. Number of ingredients in a recipe is often proportional to the complexity of the recipe, which could impact bayesian ratings, making it a relevant features.
3. 'n_steps': this is the number of steps in each recipe and is a quantitative discrete variable. Similarly, number of steps in a recipe is often proportional to the complexity of the recipe, which could impact bayesian ratings, making it a relevant features
4. 'minutes': this is the total cook time of a recipe in minutes and is quantitative continuous variable.
5. 'calories': this is the total calorie count of a recipe and is a quantitative continuous variable. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.
6. 'total_fat_PDV': this is the fat content as a percentage of the FDA daily consumption guidelines and is a quantitative continuous variable. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.
7. 'sugar_PDV':  this is the sugar content as a percentage of the FDA daily consumption guidelines and is a quantitative continuous variable. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.
8. 'sodium_PDV':  this is the sodium content as a percentage of the FDA daily consumption guidelines and is a quantitative continuous variable. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.
9. 'protein_PDV':  this is the protein content as a percentage of the FDA daily consumption guidelines and is a quantitative continuous variable. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.
10. 'saturated_fat_PDV': this is the saturated fat content as a percentage of the FDA daily consumption guidelines and is a quantitative continuous variable. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.
11. 'carbohydrates_PDV': this is the carbohydrate content as a percentage of the FDA daily consumption guidelines and is a quantitative continuous variable. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.

## Categorical Variables & Transformations

We also added 2 categorical variables: 
1. 'ingredients': this is the list of ingredients in the recipe which is a categorical nominal variable. Because important of ingredient is not weighed here, we use CountVectorizer to get a simple frequency count of ingredients. This is a component of the overall nutritional profile of a recipe. Thus, people following different diets and with varying health conditions could react differently to a recipe than others. This makes it a relevant feature to consider.
2. 'tags':  this is the list of tags for a given recipe which is a categorical nominal variable. For this variable, we wanted to weigh the tags based on rarity and frequency and TF-IDF works perfectly in this case as it compares how often a tag appears in a given recipe versus how rare it is across all recipes. Certain tags may get more visibility due to factors like current trends, diets, etc. thus would be a relevant feature to consider. 

### Cross Validation & Grid Search

To train this model we first used a Grid Search with a K-Folds Cross Validation to find the optimal hyper parameters for this model. 

We used 4 splits since our data ranges from 1 to 5

Furthermore we used \[R^2\] as our metric to compare parameter sets because of its regard for under fitting models and that is what we want our model to optimize for as well. 

To speed up processing we changed 'n_jobs' to -1 to use multicore processing

This yielded the following hyperparams:

 1. 'n_estimator': 100
 2. 'max_depth': 5
 3. 'min_samples_leaf': 20
 4. 'max_features': 'sqrt' (squareroot)

### Results

Our final model has a considerably better \[R^2\] of 0.2989. 
This is almost a 1000x increase in correlation to the real data when compared to the baseline! Astonishing! 

<iframe
  src="assets/final.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot compares the residuals of our baseline model with the actual bayesian ratings of our dataset. Although this model still has quite a way to go with consistently accurate predictions, it shows considerably less underfitting and has a better reaction to outlier variance.

<iframe
  src="assets/final_res1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

 This plot compares the residuals of our baseline model with the actual bayesian ratings of our dataset. These residuals show a clear improvement in variance. Nevertheless we can see some underfitting towards extreme values, but the model does actually adapts to features. The wider spread of residuals at the upper ratings shows mild heteroscedasticity.

<iframe
  src="assets/final_res2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

 This plot compares the residuals of our baseline model with the predicted bayesian ratings of our model. These residuals show consistent negative residuals which means it is repeatedly overpredicting from the true rating. Moreover, there are very few positive residuals, meaning it rarely undepredicts where a good model would have balanced extreme residuals. Yet, the model shows some heteroscedasticity and is a large improvement in predicting bayesian ratings when compared to our baseline.

 To conclude, this model not only significantly outperforms the baseline in regards to the \[R^2\] metric, it also shows considerably more variance in its residuals and predictions, therefore reducing the underfitting that was apparent in the baseline model, while also showing considerable improvements in learning the trends of the outliers and extreme data. 

## Fairness Analysis

### Question & Groups

We wanted to know if our model was biased towards higher or lower ratings or if it fairly predicted both groups.

Our two groups are:

Highly-rated recipes (>= 4.63 Bayesian Rating)

Lowly-rated recipes (< 4.63 Bayesian Rating)

4.63 is the 75th percentile of the list of Bayesian Ratings of all recipes, and thus allowing us to isolate the higher rated group in comparison to others. When dealing with such skewed data, it is important to see the contrast of the model's predictions with outliers and the most common group. By choosing the upper quartile as the interval for our split, we gain more insight into how fair model's accuracy when comparing the top 25th percent of recipes versus the rest. 

We would like to test the following hypotheses:

Null Hypothesis \[H_0\]: The model is fair, as the precision for recipes with higher ratings and lower ratings are equal (and any difference is due to random noise).

Alternative Hypothesis \[H_1\]: The model is unfair, as the precision, RMSE< for recipes with higher ratings and lower ratings are unequal.

### Metrics:

The metric we chose to test these hypotheses was RMSE as we were only testing for a difference, not a directional bias. Furthermore RMSE captures the magnitude of difference, which is all we need to test our hypotheses

### Significance Level

We chose a significance level of 0.01 because we wanted a statistically significant result if rejecting the null hypothesis to avoid false positive results and build robustness. 

### Results:

Observed difference (Low - High): 0.0357 
P-value: 0.0000

<iframe
  src="assets/fairness.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Conclusion

With this p-value we can reject the null hypothesis. Therefore we are lead to believe that our model has a disparity in its accuracy towards high and low rating groups and thus bias towards a certain group.

We had expected this to happen since our model is still quite sensitive to the immense bias in the data. 

To combat this, we would ideally collect more data but realistically would use a form of imputation such as litwise deletion, so each rating group has an equal number of recipes or probabilistic imputation to assume recipes for the rest.

<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>