import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from bs4 import BeautifulSoup
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

GenreAll = ['Action','Adventure','Animation','Children','Comedy',
            'Crime','Documentary','Drama','Fantasy','FilmNoir',
            'Horror','Musical','Mystery','Romance','SciFi',
            'Thriller','War','Western']

incl_year = ['Action','Adventure','Animation','Children','Comedy',
            'Crime','Documentary','Drama','Fantasy','FilmNoir',
            'Horror','Musical','Mystery','Romance','SciFi',
            'Thriller','War','Western','av_Year']

c = ['aqua',  'coral','darkred','magenta','orange',
     'olive','#800080','blue','#4682B4','mediumseagreen',
     'red','navy', 'green','khaki', 'salmon', 'teal', 
     'yellow', 'lightblue']

c_dic={}
for i in range(len(GenreAll)):
    c_dic[GenreAll[i]]=c[i]
    
def title_cleaning(df):
    
    print("Separate the year from the title using regular expressions...")
    print(df)    
    df['Title'] = df['title'].apply(lambda x:x.split(' (')[0])

    df['Year'] = df['title'].apply(lambda x:re.findall(r"\([0-9]{4,7}\)", x))
    df['Year'] = df['Year'].apply(lambda x:''.join(x))
    df['Year'] = df['Year'].str[1:5]
    
    ## how many rows without year information
    print("how many rows w/o year :  "+str(len(df[df['Year']==''])))
    return(df)
## Scrape data from IMDB website
def extract_imdb(df):
    
    print("Scrape data from IMDB website...")
    
    mid = df[df.Year==''].movieId.values
    year=[]
    num=0
    for i in mid:
        n = Link[Link.movieId==i].imdbId.values
        url = 'http://www.imdb.com/title/tt'+str(n[0])
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        title = soup.select('title')
        result = re.findall('\((.*?)\)', str(title[0]))
        if result != []:
            year.append(result[0])
            print(result[0],num)
        else:
            year.append('00')
            print(result,num)
        num+=1
    return(year)

## Organize the IMDB year information
def year_cleaning(df,imdb_year):

    print("Organize the IMDB year information......")

    df.loc[df['Year']=='','Year']=imdb_year
    df['Year'] = df['Year'].apply(lambda x:re.findall('^[0-9]*', x)[0])
    print("number of movies like TV movies and series : "+ str(len(df[df['Year']==''])))

    df.drop(df[df['Year']==''].index,inplace=True)
    df['Year'] = df['Year'].astype(int)
    
    return(df)

## genre cleaning
def genre_cleaning(df):
    
    print("cleaning genres..... ")
    
    df['genres']=df['genres'].apply(lambda x:re.sub("Sci-Fi", "SciFi", x))
    df['genres']=df['genres'].apply(lambda x:re.sub("Film-Noir", "FilmNoir", x))

    print("number of (no genres listed) : "+ str(len(df[df['genres']=='(no genres listed)'])))

#    df.drop(df[df['genres']=='(no genres listed)'].index,inplace=True)
    
    return(df)

## Make dummy variables for genres
def genre_dummies(df):

    print("Make dummy variables for genres... ")

    GenreList = ['Action','Adventure','Animation','Children','Comedy',
             'Crime','Documentary','Drama','Fantasy','FilmNoir',
             'Horror','Musical','Mystery','Romance','SciFi',
             'Thriller','War','Western','IMAX','(no genres listed)']

    GenreTable = pd.DataFrame(np.zeros((len(df),len(GenreList))),index=df.index, columns=GenreList)
    result = pd.concat([df,GenreTable],axis=1)

    for j in result.index:
        temp = result['genres'].loc[j].split('|')
        result.loc[j,temp]=1

    result.columns=[['movieId','title','genres','Title','Year',
                         'Action','Adventure','Animation','Children','Comedy',
                         'Crime','Documentary','Drama','Fantasy','FilmNoir',
                         'Horror','Musical','Mystery','Romance','SciFi',
                         'Thriller','War','Western','IMAX','(no genres listed)']]

    del result['IMAX']
    del result['(no genres listed)']
    return(result)

## Do some feature engineering, like counts of genres reviewed for each user
def data_featuring(df):

    print("Do some feature engineering, like counts of genres reviewed for each user... ")

    group_movierating = df.groupby('userId')
    temp = group_movierating.sum()[GenreAll]
    genre_dist = pd.DataFrame(temp.values/pd.DataFrame(temp.T.sum()).values,
                                columns=GenreAll,index=temp.index)
    genre_dist['TotalNumber']=group_movierating.count()['movieId']

    df[GenreAll] = df[GenreAll].apply(lambda x: x*df['rating'])
    df = df.replace(0, np.NaN)
    average_rating = df.groupby('userId').mean()
    average_rating.drop('timestamp',axis=1,inplace=True)

    result = pd.concat([genre_dist,average_rating],axis=1)
    
    result.columns=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical',
       'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western',
       'TotalNumber', 'movieId', 'av_Year', 'av_Action', 'av_Adventure', 'av_Animation',
       'av_Children', 'av_Comedy', 'av_Crime', 'av_Documentary', 'av_Drama', 'av_Fantasy',
       'av_FilmNoir', 'av_Horror', 'av_Musical', 'av_Mystery', 'av_Romance', 'av_SciFi',
       'av_Thriller', 'av_War', 'av_Western', 'av_rating']
    return(result)

## Let's plot...!!
def pie_fig(df,df_num):
    fig = plt.figure(figsize=(16,8))
    ax=fig.add_subplot(121)
    contents = df.iloc[0].sort_values(ascending=False)
    num = df_num.iloc[0]
    plt.pie(contents, labels=contents.index,
        colors=[c_dic.get(key) for key in contents.index.values],
        autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    ax.set_title("person A's movie review - Total Num. : "+str(num),fontsize=16)

    ax=fig.add_subplot(122)
    contents = df.iloc[3].sort_values(ascending=False)
    num = df_num.iloc[3]
    plt.pie(contents, labels=contents.index,
        colors=[c_dic.get(key) for key in contents.index.values],
        autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    ax.set_title("person B's movie review - Total Num. :"+str(num),fontsize=16)

def pie_fig_horror(df0, df,df_num):
    fig = plt.figure(figsize=(16,8))
    ax=fig.add_subplot(121)
    idnum = df.sort_values('Horror',ascending=False).index[0]
    contents = df0[df0.index==idnum][GenreAll].iloc[0].sort_values(ascending=False)
    num = df0[df0.index==idnum]['TotalNumber'].iloc[0]
    plt.pie(contents, labels=contents.index,
        colors=[c_dic.get(key) for key in contents.index.values],
        autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    ax.set_title("Horror fan's movie review - Total Num. : "+str(num),fontsize=16)
    
    ax=fig.add_subplot(122)
    idnum = df.sort_values('Horror',ascending=False).index[1]
    contents = df0[df0.index==idnum][GenreAll].iloc[0].sort_values(ascending=False)
    num = df0[df0.index==idnum]['TotalNumber'].iloc[0]
    plt.pie(contents, labels=contents.index,
        colors=[c_dic.get(key) for key in contents.index.values],
        autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    ax.set_title("Horror fan's movie review - Total Num. : "+str(num),fontsize=16)


## Let's find the most popular movie for each genre
def top30_find(df,MovieRating):
    for j in np.arange(len(GenreAll)):
        name1='Group1'+str(GenreAll[j])
        name2='Group2'+str(GenreAll[j])
        exec(name1+"= pd.DataFrame(df.sort_values(GenreAll[j],ascending=False).head(500).index)")
        exec(name2+"= pd.DataFrame(df.sort_values(GenreAll[j],ascending=False).tail(49500).index)")
        exec(name1+".columns=['userId']")
        exec(name2+".columns=['userId']")
    
    title_sum = MovieRating.groupby('title').sum()
    for j in np.arange(len(GenreAll)):
        temp = title_sum.sort_values(GenreAll[j],ascending=False)
        exec('Result'+str(GenreAll[j])+'=DataFrame()')
        i=0; n=0
        while n<=30:
            name = temp.index[i]
            exec('G1 = pd.merge(Group1'+str(GenreAll[j])+''',MovieRating[MovieRating['title']==name][['userId','rating']],on='userId',how='inner')''')
            if len(G1) >=100:
                exec('G2 = pd.merge(Group2'+str(GenreAll[j])+''',MovieRating[MovieRating['title']==name][['userId','rating']],on='userId',how='inner')''') 
                print(i, n, GenreAll[j],name,len(G1),round(G1['rating'].mean(),2),round(G1['rating'].std(),2),len(G2),round(G2['rating'].mean(),2),round(G2['rating'].std(),2))
                exec('Result'+str(GenreAll[j])+'''[n]=[GenreAll[j],name,len(G1),round(G1['rating'].mean(),2),round(G1['rating'].std(),2),len(G2),round(G2['rating'].mean(),2),round(G2['rating'].std(),2)]''')
                n+=1
            i+=1
            if i>100:
                break
        print(GenreAll[j])

    for i in range(len(GenreAll)):
        Genre = GenreAll[i]
        exec('Result1 = Result'+str(Genre)+'.T')
        Result1.T.to_csv(str(Genre)+'.csv')
        
## Let's plot the most popular movies ..!
def top30_cleaning(top30,name):
    
    top30.drop('Unnamed: 0',inplace=True)
    top30.columns=[['Genre','title','NumFan','RatingFan','StdFan','NumPeo','RatingPeo','StdPeo']]
    top30.columns = top30.columns.get_level_values(0)
    top30['RatingFan']=top30['RatingFan'].apply(float)
    top30['StdFan']=top30['StdFan'].apply(float)
    top30['RatingPeo']=top30['RatingPeo'].apply(float)
    top30['StdPeo']=top30['StdPeo'].apply(float)
    top30['NumFan']=top30['NumFan'].apply(int)
    top30['NumPeo']=top30['NumPeo'].apply(int)
    top30['Diff']=top30['RatingFan']-top30['RatingPeo']
    top30 = top30.set_index('title')    
    return(top30)

def top30_plot(top30,name):

    top30= top30_cleaning(top30,name)
    top30_ascending = top30.sort_values('Diff',ascending=True)
    
    pos_diff = top30_ascending[top30_ascending['Diff']>=0]
    neg_diff = top30_ascending[top30_ascending['Diff']<0]
    
    x0 = np.arange(len(pos_diff))
    x1 = np.arange(len(neg_diff))

    binsize=0.3
    f = plt.figure(figsize=(5,10))

    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.barh(x0+len(x1)+0.36, pos_diff['RatingFan'], binsize,color='lightcoral')
    ax.barh(x0+len(x1)+0.18, pos_diff['RatingPeo'], binsize,color='deepskyblue')
    ax.barh(x0+len(x1), pos_diff['Diff'], binsize+0.2,color='blue')

    ax.barh(x1+0.36, neg_diff['RatingFan'], binsize,color='lightcoral')
    ax.barh(x1+0.18, neg_diff['RatingPeo'], binsize,color='deepskyblue')
    ax.barh(x1, neg_diff['Diff'], binsize+0.2,color='red')

    plt.title(name,size=16)
    plt.xlabel('Rating',fontsize=14)
    xx = np.zeros(len(top30))
    yy = np.arange(len(top30))+0.1

    ax.set_yticks(np.arange(len(top30))+0.3)
    ax.set_yticklabels(top30_ascending.index,rotation=0)
    plt.text(5,3.3,'Rating(Fan)',fontsize=16,color='tomato')

    plt.text(5,2.2,'Rating(Others)',fontsize=16,color='dodgerblue')
    plt.text(5,1.1,'Rating(Fan)-Rating(Others) (+)',fontsize=16,color='darkblue')
    plt.text(5,0.,'Rating(Fan)-Rating(Others) (-)',fontsize=16,color='red')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    f.show()

def overall_plot(MovieGenre):
    d = ['Act', 'Adv', 'Anim','Child','Com','Cri','Doc','Dra',
     'Fant','Noir','Horr','Mus','Myst','Rom','SF','Thril','War','West']
    reorder=[1,3,5,7,9,11,13,15,17,2,4,6,8,10,12,14,16,18]

    fig = plt.figure(figsize=(35,20))
    plt.title("Overlaps between ( ) and other genres",fontsize=30)
    for j in range(len(GenreAll)):
        clc = ['plum','skyblue','olive','lightcoral','silver','lightsalmon',
              'darkkhaki','lightsteelblue','cyan','lightpink','peru','gold',
              'darkcyan','darksalmon','yellowgreen','orange','thistle','cadetblue']
        theMovie = MovieGenre[MovieGenre[GenreAll[j]]==1.0]
        a=[]
        for i in range(len(GenreAll)):
            n = len(theMovie[theMovie[GenreAll[i]]==1.0])
            a.append(n)
        frame = pd.DataFrame(GenreAll,a).reset_index()
        frame.columns=['y','x']
        ax = fig.add_subplot(len(GenreAll)/2,2,reorder[j])
        clc[j]='brown'
        sns.barplot(frame.x,frame.y,palette=clc)
        ax.set_xticklabels(d, rotation=0, fontsize=20)
#    plt.subplots_adjust(hspace=0.3,wspace=0.2)
#    plt.savefig('testplot.png')

## Do preprocessing for PCA plot
def pca_plot(df,n):

    print("Do preprocessing for PCA plot... ")
        
    scaler = MinMaxScaler()
    scaler.fit(df)
    final_top_preprocessing = scaler.transform(df)
    pca = PCA(n_components=n).fit(final_top_preprocessing)

    dimensions = ['Dim {}'.format(i) for i in range(1,len(pca.components_)+1)]

    components = pd.DataFrame(np.round(pca.components_, 4), columns = df.keys())
    components.index = dimensions

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    fig,ax = plt.subplots(figsize = (20,8))

    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights", fontsize=18)
    ax.legend_.remove()
    ax.set_xticklabels(dimensions, rotation=0, fontsize=18)
    fig.legend(loc='lower left',fontsize=14)

    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev),fontsize=16)

## Apply PCA
def pca_function(df,n):
    
    print("Apply PCA... ")
    
    scaler = MinMaxScaler()
    scaler.fit(df)
    final_top_preprocessing = scaler.transform(df)
    
    pca = PCA(n_components=n,random_state=40).fit(final_top_preprocessing)
    reduced_data = pca.transform(final_top_preprocessing)
    reduced_data = pd.DataFrame(reduced_data)
    co = ['']*n
    for i in range(n):
        co[i]='Dim '+str(i+1)
    reduced_data.columns=co
    return(reduced_data)

## Kmeans Clustering with various K values
def check_kmean(reduced_data,n):

    print("Kmeans Clustering with various K values... ")

    for i in range(2,n+1):
        kmeans = KMeans(n_clusters=i,random_state=0).fit(reduced_data)
        preds = kmeans.predict(reduced_data)
        centers = kmeans.cluster_centers_
        score = silhouette_score(reduced_data,preds)
        print(i,round(score,2))
    
def result_pca_plot(result_pca,movie_name,name):

    print(result_pca[result_pca.title==movie_name]['genre'].unique())
    
    plt.figure(figsize=(13,6))
    a = result_pca[result_pca.title==movie_name]
    ax= sns.barplot(x="group",y="mean",data=a[a.genre==name])
    ax.set_xticklabels(['G1','G2','G3','G4','G5','G6'], rotation=0, fontsize=32)
    fontsize=32
    plt.ylim(2.5,4.8)
    plt.title(movie_name,fontsize=32)
    plt.xlabel("", fontsize=30)
    plt.ylabel("Rating (mean)",fontsize=fontsize)
    plt.text(-0.35,3,'Cri',fontsize=fontsize,color='black')
    plt.text(-0.35,2.8,'Thri',fontsize=fontsize,color='k')
    plt.text(-0.35,2.6,'Myst',fontsize=fontsize,color='k')

    plt.text(0.7,3,'Act',fontsize=fontsize,color='darkred')
    plt.text(0.7,2.8,'Adv',fontsize=fontsize,color='darkred')
    plt.text(0.7,2.6,'SF',fontsize=fontsize,color='darkred')

    plt.text(1.7,3,'Com',fontsize=fontsize,color='darkslategrey')
    plt.text(1.7,2.8,'Dra',fontsize=fontsize,color='darkslategrey')
    plt.text(1.7,2.6,'Rom',fontsize=fontsize,color='darkslategrey')

    plt.text(3.7,3.2,'Adv',fontsize=fontsize,color='white')
    plt.text(3.7,3,'Ani',fontsize=fontsize,color='white')
    plt.text(3.7,2.8,'Child',fontsize=fontsize,color='white')
    plt.text(3.7,2.6,'Fant',fontsize=fontsize,color='white')

    plt.text(4.7,3.,'Dra',fontsize=fontsize,color='white')
    plt.text(4.7,2.8,'Doc',fontsize=fontsize,color='white')
    plt.text(4.7,2.6,'Noir',fontsize=fontsize,color='white')

