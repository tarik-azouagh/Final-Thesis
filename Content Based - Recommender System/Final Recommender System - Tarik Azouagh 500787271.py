# %% [markdown]
# ### Cleaning scraped AH Chocolatebar data
# 
# The goal is to clean the data to make a system/machine that can:
# 1) Rate each individual chocolatebar based on their sustainability certificate. (The ratings are substracted from Milieu Centraal).
# 2) When selecting a chocolatebar: Recommend a different chocolatebar that is sustainably more responsible with the same characteristics (based on their ingredients).
# 
# I.E. when scanning a Lindt dark chocolate with hazelnut (No Certificate), the system returns: the certificates of the bar, a rating of the certificates, the raw materials used, 
# and a recommendation for a more sustainable (dark) chocolate bar (with hazelnut).

# %%
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# %%
#df = pd.read_json(r'C:\Users\Tarik\Desktop\Master Digital Driven Business\choco1.json')
df = pd.read_json(r'C:\Users\Tarik\Desktop\Master Digital Driven Business\choco-AH.json')

# %%
#Turn prijs, kilo_prijs, inhoud_gewicht, omschrijving, product_naam, product_id, and ingredienten into readable string variable
df['prijs'] = [','.join(map(str, l)) for l in df['prijs']]
df['prijs'] = df['prijs'].replace(',','', regex=True)
df['kilo_prijs'] = df['kilo_prijs'].str[-1]
df['kilo_prijs'] = df['kilo_prijs'].str.replace(',', '.')
df['inhoud_gewicht'] = [','.join(map(str, l)) for l in df['inhoud_gewicht']]
df['omschrijving'] = [','.join(map(str, l)) for l in df['omschrijving']]
df['product_naam'] = [','.join(map(str, l)) for l in df['product_naam']]
df['product_id'] = [','.join(map(str, l)) for l in df['product_id']]
df['ingredienten'] = [','.join(map(str, l)) for l in df['ingredienten']]

# %%
#replace empty strings with missing values
df['product_id']=df['product_id'].str.strip().replace('',np.nan)
df['product_id']=df['product_id'].str.strip().replace(' ',np.nan)
df['product_naam']=df['product_naam'].str.strip().replace('',np.nan)
df['prijs']=df['prijs'].str.strip().replace('',np.nan)
df['kilo_prijs'] = df['kilo_prijs'].str.strip().replace('',np.nan)
df['omschrijving']=df['omschrijving'].str.strip().replace('',np.nan)
df['inhoud_gewicht'] = df['inhoud_gewicht'].str.strip().replace('',np.nan)
df['ingredienten']=df['ingredienten'].str.strip().replace('',np.nan)

# %%
#drop rows missing a product_id
df = df.dropna(subset=['product_id'])
df = df.reset_index(drop=True)

# %%
#turn prijs and kilo_prijs into decimal values
df['prijs'] = df['prijs'].astype(float)
df['kilo_prijs'] = df['kilo_prijs'].astype(float)

# %% [markdown]
# #### Getting all unique values out of the kenmerken column
# 
# - Converting the lists in the column in a way that the computer can easily read the individual kenmerken (certificates)

# %%
#Making dummies out of relavant features in the kenmerken column
kenmerken = pd.get_dummies(df['kenmerken'].apply(pd.Series).stack(dropna=False)).sum(level=0)
kenmerken['Rainforest Alliance'] = kenmerken['Rainforest Alliance'] + kenmerken['Rainforest Alliance people & nature']
kenmerken.drop('Rainforest Alliance people & nature', axis=1, inplace=True)
kenmerken.drop('Vegetarisch', axis=1, inplace=True)
kenmerken.drop('Glutenvrij', axis=1, inplace=True)
kenmerken.drop('Melkvrij', axis=1, inplace=True)
kenmerken.drop('Groene Punt', axis=1, inplace=True)
kenmerken.drop('Triman', axis=1, inplace=True)
kenmerken.drop('Society of the Plastics Industry (SPI)', axis=1, inplace=True)
kenmerken.drop('Sustainable Palm Oil RSPO Certified', axis=1, inplace=True)
kenmerken.drop('Recyclebaar', axis=1, inplace=True)
kenmerken.drop('Veganistisch', axis=1, inplace=True)

# %%
#join df and kenmerken dataframes on index
df1 = pd.merge(df, kenmerken, left_index=True, right_index=True)

# %%
# Insert the Milieu Centraal ratings
mc = {'name': ['UTZ', 'Rainforest Alliance', 'Fairtrade', 'Fair for Life', 'Climate Neutral Certified', 'Biologisch', 'EKO', 'Demeter', 'Cocoa Horizons', 'Cocoa Life', "Tony's Open Chain"],
        'environment': [5, 5, 3, 3, 3, 3, 3, 3, 2, 0, 0],
        'social': [5, 5, 5, 5, 0, 0, 0, 2, 1, 0, 0],
        'control': [5, 5, 5, 4, 4, 4, 4, 4, 1, 0, 0],
        'transparency': [5, 5, 4, 4, 4, 4, 5, 4, 2, 0, 5]
        }
mc = pd.DataFrame.from_dict(mc)


# %%
#calculate avg score social
mc['avg_social'] = np.where(mc.social>0, (mc['social'] + mc['control'] + mc['transparency']) / 3, 0)
mc['avg_social'] = np.where(mc.name=="Tony's Open Chain", (mc['social'] + mc['control'] + mc['transparency']) / 3, mc.avg_social)

# %%
#calculate avg score environment
mc['avg_environment'] = np.where(mc.environment>0, (mc['environment'] + mc['control'] + mc['transparency']) / 3, 0)
#mc['avg_environment'] = np.where(mc.name=="Tony's Open Chain", (mc['social'] + mc['control'] + mc['transparency']) / 3, mc.avg_social)    

# %%
#split social and environmental binary values into two seperate columns per certificate
df1["UTZ_social"] = df1["UTZ"]
df1["Biologisch_social"] = df1["Biologisch"]
df1["Cocoa_Horizons_social"] = df1["Cocoa Horizons"]
df1["Cocoa_Life_social"] = df1["Cocoa Life"]
df1["Fairtrade_social"] = df1["Fairtrade"]
df1["Rainforest_Alliance_social"] = df1["Rainforest Alliance"]
df1["Tonys_Open_Chain_social"] = df1["Tony's Open Chain"]

df1["UTZ_environment"] = df1["UTZ"]
df1["Biologisch_environment"] = df1["Biologisch"]
df1["Cocoa_Horizons_environment"] = df1["Cocoa Horizons"]
df1["Cocoa_Life_environment"] = df1["Cocoa Life"]
df1["Fairtrade_environment"] = df1["Fairtrade"]
df1["Rainforest_Alliance_environment"] = df1["Rainforest Alliance"]
df1["Tonys_Open_Chain_environment"] = df1["Tony's Open Chain"]

# %%
#Insert overall ratingsvalue into the certificate columns in the df1 frame
df1['UTZ_social'] = np.where(df1["UTZ"]==1, mc.loc[mc['name'] == "UTZ", 'avg_social'].item(), 0)
df1['Biologisch_social'] = np.where(df1["Biologisch"]==1, mc.loc[mc['name'] == "Biologisch", 'avg_social'].item(), 0)
df1['Cocoa_Horizons_social'] = np.where(df1["Cocoa Horizons"]==1, mc.loc[mc['name'] == "Cocoa Horizons", 'avg_social'].item(), 0)
df1['Cocoa_Life_social'] = np.where(df1["Cocoa Life"]==1, mc.loc[mc['name'] == "Cocoa Life", 'avg_social'].item(), 0)
df1['Fairtrade_social'] = np.where(df1["Fairtrade"]==1, mc.loc[mc['name'] == "Fairtrade", 'avg_social'].item(), 0)
df1['Rainforest_Alliance_social'] = np.where(df1["Rainforest Alliance"]==1, mc.loc[mc['name'] == "Rainforest Alliance", 'avg_social'].item(), 0)
df1['Tonys_Open_Chain_social'] = np.where(df1["Tony's Open Chain"]==1, mc.loc[mc['name'] == "Tony's Open Chain", 'avg_social'].item(), 0)

df1['UTZ_environment'] = np.where(df1["UTZ"]==1, mc.loc[mc['name'] == "UTZ", 'avg_environment'].item(), 0)
df1['Biologisch_environment'] = np.where(df1["Biologisch"]==1, mc.loc[mc['name'] == "Biologisch", 'avg_environment'].item(), 0)
df1['Cocoa_Horizons_environment'] = np.where(df1["Cocoa Horizons"]==1, mc.loc[mc['name'] == "Cocoa Horizons", 'avg_environment'].item(), 0)
df1['Cocoa_Life_environment'] = np.where(df1["Cocoa Life"]==1, mc.loc[mc['name'] == "Cocoa Life", 'avg_environment'].item(), 0)
df1['Fairtrade_environment'] = np.where(df1["Fairtrade"]==1, mc.loc[mc['name'] == "Fairtrade", 'avg_environment'].item(), 0)
df1['Rainforest_Alliance_environment'] = np.where(df1["Rainforest Alliance"]==1, mc.loc[mc['name'] == "Rainforest Alliance", 'avg_environment'].item(), 0)
df1['Tonys_Open_Chain_environment'] = np.where(df1["Tony's Open Chain"]==1, mc.loc[mc['name'] == "Tony's Open Chain", 'avg_environment'].item(), 0)

# %%
#Round the certificate scores to two decimals
df1["UTZ_social"] = df1["UTZ_social"].round(decimals=2)
df1["Biologisch_social"] = df1["Biologisch_social"].round(decimals=2)
df1["Cocoa_Horizons_social"] = df1["Cocoa_Horizons_social"].round(decimals=2)
df1["Cocoa_Life_social"] = df1["Cocoa_Life_social"].round(decimals=2)
df1["Fairtrade_social"] = df1["Fairtrade_social"].round(decimals=2)
df1["Rainforest_Alliance_social"] = df1["Rainforest_Alliance_social"].round(decimals=2)
df1["Tonys_Open_Chain_social"] = df1["Tonys_Open_Chain_social"].round(decimals=2)

df1["UTZ_environment"] = df1["UTZ_environment"].round(decimals=2)
df1["Biologisch_environment"] = df1["Biologisch_environment"].round(decimals=2)
df1["Cocoa_Horizons_environment"] = df1["Cocoa_Horizons_environment"].round(decimals=2)
df1["Cocoa_Life_environment"] = df1["Cocoa_Life_environment"].round(decimals=2)
df1["Fairtrade_environment"] = df1["Fairtrade_environment"].round(decimals=2)
df1["Rainforest_Alliance_environment"] = df1["Rainforest_Alliance_environment"].round(decimals=2)
df1["Tonys_Open_Chain_environment"] = df1["Tonys_Open_Chain_environment"].round(decimals=2)

# %%
#Calculate overal social & environmental rating
df1["social_rating"] = (df1["UTZ_social"] + df1["Biologisch_social"] + df1["Cocoa_Horizons_social"] + df1["Cocoa_Life_social"] + df1["Fairtrade_social"] + df1["Rainforest_Alliance_social"]
 + df1["Tonys_Open_Chain_social"])

df1["environment_rating"] = (df1["UTZ_environment"] + df1["Biologisch_environment"] + df1["Cocoa_Horizons_environment"] + df1["Cocoa_Life_environment"] + df1["Fairtrade_environment"]
 + df1["Rainforest_Alliance_environment"] + df1["Tonys_Open_Chain_environment"])

# %% [markdown]
# #### Clean ingredienten column. turn into lists and build content based recommendation system
# 

# %%
#Clean ingredienten string
df1['ingredienten'] = df1['ingredienten'].str.lower()
df1['ingredienten'] = df1['ingredienten'].str.replace('<span class="product-info-ingredients_containsallergen__1slys">', "")
df1['ingredienten'] = df1['ingredienten'].str.replace('<span>', "")
df1['ingredienten'] = df1['ingredienten'].str.replace('</span>', "")
df1['ingredienten'] = df1['ingredienten'].str.replace("ingrediënten:", "")
df1['ingredienten'] = df1['ingredienten'].str.replace("ingrediënten", "")
df1['ingredienten'] = df1['ingredienten'].str.replace("\d+", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("%", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("rainforest alliance gecertificeerd. zie voor meer informatie ra.org", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("rainforest alliance gecertificeerd. lees meer op ra.org", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("rainforest alliance gecertificeerd. www.ra.org.", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("visit info.fairtrade.net/sourcing", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("¹", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("ngrediënten:", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("www.ra.org", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("ra.org", '')
df1['ingredienten'] = df1['ingredienten'].str.replace("'", '')
df1['ingredienten'] = df1['ingredienten'].str.replace(":", '')
df1['ingredienten'] = df1['ingredienten'].str.replace(";", '')
df1['ingredienten'] = df1['ingredienten'].str.replace(".", ' ')
#df1['ingredienten'] = df1['ingredienten'].str.replace(",", ' ')
df1['ingredienten'] = df1['ingredienten'].str.replace(" ", ',')
df1['ingredienten'] = df1['ingredienten'].str.replace("*", "", regex=True)
df1['ingredienten'] = df1['ingredienten'].str.replace("°", "", regex=True)
df1['ingredienten'] = df1['ingredienten'].str.replace("<br>", "", regex=True)
df1['ingredienten'] = df1['ingredienten'].str.replace(r"\(.*\)","", regex=True)

# %%
#Clean omschrijvingen string
df1['omschrijving'] = df1['omschrijving'].str.lower()
df1['omschrijving'] = df1['omschrijving'].str.replace('<span class="product-info-ingredients_containsallergen__1slys">', "")
df1['omschrijving'] = df1['omschrijving'].str.replace('<span>', "")
df1['omschrijving'] = df1['omschrijving'].str.replace('</span>', "")
df1['omschrijving'] = df1['omschrijving'].str.replace("ingrediënten:", "")
df1['omschrijving'] = df1['omschrijving'].str.replace("ingrediënten", "")
df1['omschrijving'] = df1['omschrijving'].str.replace("\d+", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("%", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("rainforest alliance gecertificeerd. zie voor meer informatie ra.org", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("rainforest alliance gecertificeerd. lees meer op ra.org", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("rainforest alliance gecertificeerd. www.ra.org.", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("visit info.fairtrade.net/sourcing", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("¹", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("ngrediënten:", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("www.ra.org", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("ra.org", '')
df1['omschrijving'] = df1['omschrijving'].str.replace("'", '')
df1['omschrijving'] = df1['omschrijving'].str.replace(".", ' ')
df1['omschrijving'] = df1['omschrijving'].str.replace(",", ' ')
df1['omschrijving'] = df1['omschrijving'].str.replace(":", ' ')
df1['omschrijving'] = df1['omschrijving'].str.replace(";", ' ')
df1['omschrijving'] = df1['omschrijving'].str.replace(" ", ',')
df1['omschrijving'] = df1['omschrijving'].str.replace("gecertificeerte", ' ')
df1['omschrijving'] = df1['omschrijving'].str.replace("gecertificeerte", ' ')
df1['omschrijving'] = df1['omschrijving'].str.replace("*", "", regex=True)
df1['omschrijving'] = df1['omschrijving'].str.replace("°", "", regex=True)
df1['omschrijving'] = df1['omschrijving'].str.replace("<br>", "", regex=True)
df1['omschrijving'] = df1['omschrijving'].str.replace(r"\(.*\)","", regex=True)

# %%
#Make new data frame out of relevant columns for the recommender system
df = df1[['product_naam', 'omschrijving', 'ingredienten', 'kenmerken', 'social_rating','environment_rating']]

# %% [markdown]
# ### Recommender System

# %%
#Place a index in the data frame
df.reset_index(inplace=True)

# %%
#Scaling the social_ratings and environment_ratings from 0 to 1
a, b = 0, 1
x, y = df[['social_rating', 'environment_rating']].min(), df[['social_rating', 'environment_rating']].max()
df[['social_rating', 'environment_rating']] = (df[['social_rating', 'environment_rating']] - x) / (y - x) * (b - a) + a

# %%
#Manually make a object containing the stop_words
stop_word = ["aan","aangaande","aangezien","achte","achter","achterna","af","afgelopen","al","aldaar","aldus","alhoewel","alias","alle","allebei","alleen","alles","als","alsnog","altijd",
"altoos","ander","andere","anders","anderszins","beetje","behalve","behoudens","beide","beiden","ben","beneden","bent","bepaald","betreffende","bij","bijna","bijv","binnen","binnenin",
"blijkbaar","blijken","boven","bovenal","bovendien","bovengenoemd","bovenstaand","bovenvermeld","buiten","bv","daar","daardoor","daarheen","daarin","daarna","daarnet","daarom","daarop",
"daaruit","daarvanlangs","dan","dat","de","deden","deed","der","derde","derhalve","dertig","deze","dhr","die","dikwijls","dit","doch","doe","doen","doet","door","doorgaand","drie","duizend",
"dus","echter","een","eens","eer","eerdat","eerder","eerlang","eerst","eerste","eigen","eigenlijk","elk","elke","en","enig","enige","enigszins","enkel","er","erdoor","erg","ergens","etc",
"etcetera","even","eveneens","evenwel","gauw","ge","gedurende","geen","gehad","gekund","geleden","gelijk","gemoeten","gemogen","genoeg","geweest","gewoon","gewoonweg","haar","haarzelf","had",
"hadden","hare","heb","hebben","hebt","hedden","heeft","heel","hem","hemzelf","hen","het","hetzelfde","hier","hierbeneden","hierboven","hierin","hierna","hierom","hij","hijzelf","hoe","hoewel",
"honderd","hun","hunne","ieder","iedere","iedereen","iemand","iets","ik","ikzelf","in","inderdaad","inmiddels","intussen","inzake","is","ja","je","jezelf","jij","jijzelf","jou","jouw","jouwe",
"juist","jullie","kan","klaar","kon","konden","krachtens","kun","kunnen","kunt","laatst","later","liever","lijken","lijkt","maak","maakt","maakte","maakten","maar","mag","maken","me","meer",
"meest","meestal","men","met","mevr","mezelf","mij","mijn","mijnent","mijner","mijzelf","minder","miss","misschien","missen","mits","mocht","mochten","moest","moesten","moet","moeten","mogen",
"mr","mrs","mw","na","naar","nadat","nam","namelijk","nee","neem","negen","nemen","nergens","net","niemand","niet","niets","niks","noch","nochtans","nog","nogal","nooit","nu","nv","of","ofschoon",
"om","omdat","omhoog","omlaag","omstreeks","omtrent","omver","ondanks","onder","ondertussen","ongeveer","ons","onszelf","onze","onzeker","ooit","ook","op","opnieuw","opzij","over","overal","overeind",
"overige","overigens","paar","pas","per","precies","recent","redelijk","reeds","rond","rondom","samen","sedert","sinds","sindsdien","slechts","sommige","spoedig","steeds","tamelijk","te","tegen","tegenover",
"tenzij","terwijl","thans","tien","tiende","tijdens","tja","toch","toe","toen","toenmaals","toenmalig","tot","totdat","tussen","twee","tweede","u","uit","uitgezonderd","uw","vaak","vaakwat","van","vanaf","vandaan",
"vanuit","vanwege","veel","veeleer","veertig","verder","verscheidene","verschillende","vervolgens","via","vier","vierde","vijf","vijfde","vijftig","vol","volgend","volgens","voor","vooraf","vooral","vooralsnog",
"voorbij","voordat","voordezen","voordien","voorheen","voorop","voorts","vooruit","vrij","vroeg","waar","waarom","waarschijnlijk","wanneer","want","waren","was","wat","we","wederom","weer","weg","wegens","weinig",
"wel","weldra","welk","welke","werd","werden","werder","wezen","whatever","wie","wiens","wier","wij","wijzelf","wil","wilden","willen","word","worden","wordt","zal","ze","zei","zeker","zelf","zelfde","zelfs","zes",
"zeven","zich","zichzelf","zij","zijn","zijne","zijzelf","zo","zoals","zodat","zodra","zonder","zou","zouden","zowat","zulk","zulke","zullen","zult",'g','kg','mg','cm','p','per','l','cl','ml','ten','minste','gram',
'kilo','kilogram','millie','milliegram','centi','centigram', 'liter', 'andere', 'bevatten', 'kan', 'landbouw', '  '
'sporen', 'uit', 'sporen', 'rainforest', 'alliance', 'fairtrade', 'gecertificeerd', 'certificeren', 'certificaat', ':', ';', 'mogelijk', 'bevat', 'voor', 'meer', 'informatie', 'ra.org', '', 'waarvan', "tony's chocolonely",
"ritter sport","côte d'or","delicata","extra","fijne","fijn","lekker","lekkere","open","chain","opgezet","www.tonysopenchain.com","samenwerkingsprincipes","werken","werkt","lees","meer","minder","op", "onder", "heerlijk",
"door", "cocoa life", "creeert", "krachtig", "onmiskenbare", "Cote dor", "bouwt", "zorgvuldige", "selectie", "ongerept", "dankt", "cocoa horizons", "toegevoegde", "toegevoegd", "reep", 'milka', 'nestlé', 'nestle',
]

# %%
#Fill missing features with '' so the machine will not give problems if there are no values
features = ['ingredienten', 'omschrijving']
for feature in features:
    df[feature] = df[feature].fillna('')

# %%
#combine the features into one text and make a seperate column
def combined_features(row):
    return row['omschrijving']+" "+row['ingredienten'] 
df["combined_features"] = df.apply(combined_features, axis =1)

# %%
#Make a list from the combined features, remove all stopwords and empty strings, and return back into a string value
combined_features1 = df["combined_features"]
combined_features1 = df["combined_features"].str.split(',')
combined_features1 = combined_features1.values.tolist()

def check_about(lists:list):
    for i,j in enumerate(lists):
        if isinstance(j,list):
            check_about(j)
        else:
            lists[i]=lists[i].strip(' ')
    return lists
combined_features1 = check_about(combined_features1)

combined_features1 = [[x for x in y if x not in stop_word] for y in combined_features1]

df["combined_features"] = combined_features1

df["combined_features"] = df["combined_features"].apply(lambda x: ' '.join(x))

# %%
#Use CountVectorizer function to count all occurences of a certain word
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
#print("Count Matrix:", count_matrix.toarray())

# %%
#Apply a cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# %%
#Assign social rating en environment rating to an object
social_rating = df.social_rating 
environment_rating = df.environment_rating

# %%
#Make the system recognize the inserted productname (and other chocolate bars) by their index
## Insert the name of the chocolate bar you want the recommender system to work on in customers_chocolate
customers_chocolate = "Verkade Tablet melk hazelnoot"
def get_index_from_title(title):
    return df[df.product_naam == title]['index'].values[0]
chocolate_index = get_index_from_title(customers_chocolate)

# %%
#Make the similarity equation here. The weights can be adjusted.
similar_chocolate = list(enumerate(((cosine_sim[chocolate_index]*0.4) + (social_rating*0.3) + (environment_rating*0.3))))
df['similarity_score'] = cosine_sim[chocolate_index]

# %%
#Sort rows based on similarity (from high to low)
sorted_similar_chocolate = sorted(similar_chocolate, key=lambda x:x[1], reverse=True)

# %%
#Get the top 10 recommendations
def get_title_from_index(index):
    return df[df.index == index][["product_naam", 'social_rating', 'environment_rating', 'similarity_score', 'kenmerken']].values[0]
i=0
for chocolate in sorted_similar_chocolate:
    print('Chocolate = ', get_title_from_index(chocolate[0]))
    i=i+1
    if i>10:
        break


