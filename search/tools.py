import numpy as np
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix,csr_matrix,csc_matrix,save_npz,load_npz
import json
import re
from collections import Counter
import string
from gensim.models import KeyedVectors
import time
import sqlite3
import urllib.request
import lxml.etree as ET
import asyncio

# this file contains all the functions used in the search/views.py file

# stop words to forget when analysing a sentence
stop_words=["a","abord","absolument","afin","ah","ai","aie","aient","aies","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aucuns","aujourd","aujourd'hui","aupres","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","bon","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","celà","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","devrait","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","dos","douze","douzième","dring","droite","du","duquel","durant","dès","début","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","essai","est","et","etant","etc","etre","eu","eue","eues","euh","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eux","eux-mêmes","exactement","excepté","extenso","exterieur","eûmes","eût","eûtes","f","fais","faisaient","faisant","fait","faites","façon","feront","fi","flac","floc","fois","font","force","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fûmes","fût","fûtes","g","gens","h","ha","haut","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","mine","minimale","moi","moi-meme","moi-même","moindres","moins","mon","mot","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","nommés","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nouveaux","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parole","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","personnes","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","pièce","plein","plouf","plupart","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu'un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soient","sois","soit","soixante","sommes","son","sont","sous","souvent","soyez","soyons","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","sujet","superpose","sur","surtout","t","ta","tac","tandis","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","valeur","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voie","voient","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","état","étiez","étions","été","étée","étées","étés","êtes","être","ô"]
stop_words+=["»","«","''"," ","–"]
stop_words=set(stop_words)
table = str.maketrans(string.punctuation, ' '*len(string.punctuation))

#Path to the repo containing the databsase and json files
path="/home/gabriel/Documents/MPRI/Web_Data_Management/search_engine/Database/"
C = load_npz(path+"C.npz")
C = C.tocsc()

#vocabulary, each word is map to a frequency
f = open(path+'voc_dict.json')
voc_dict = json.load(f)
f.close()

# Word2Vec (w2v) model 
# it contains a bag of words, where each word is maped to a vector in dimension 200
model= KeyedVectors.load_word2vec_format(path+'frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin', binary=True)

def distance(a,b):
    return round(np.sum((a-b)**2),2)

# takes as input a list of articles ids and the w2v vector of the query
# computes the w2v distance
def get_word2vec_distances(id_list,query_vector):
    conn = sqlite3.connect(path+'word2vec.db')
    c = conn.cursor()
    distance_list=[]
    vector_list=[]
    for i in id_list:
        print(i)
        c.execute("Select * from Vectors where ID="+str(i)+";")
        result=np.array(c.fetchone()[1:]).reshape(-1,1)
        vector_list.append(result)
        distance_list.append(distance(result,query_vector))
    conn.close()
    return distance_list,vector_list

# takes as input a list of words
# computes the w2v vector of the query by doing the mean over the vector of each word
def get_vector_word2vec(text_split):
    filtered_sentence = [model.wv[w.lower()] for w in text_split if (not w in stop_words and model.wv.__contains__(w.lower()))]
    return np.mean(np.array(filtered_sentence),axis=0).reshape(-1,1)

# takes as input a text
# removes tag (links) from the wikipedia text article
def remove_tag(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    return cleantext

# takes as input a text
# clean the text by lowering all caracters, removing stop word, removing digits
# returns a list of important words
def clean_text(text):
    text_without_tag=remove_tag(text)
    text_without_tag=''.join([i for i in text_without_tag if not i.isdigit()])
    text_split=text_without_tag.translate(table).lower().replace('\n', ' ').split(' ')
    text_without_tag=[i for i in text_split if i not in stop_words and i!=""]
    return text_without_tag

# takes as input a text
# returns 2 dict of words, and bigrams with their corresponding frequency into the text
# ex : {aller:1,manger:4,}
def get_counters(text):
    text_without_tag=clean_text(text)
    bigrams=[text_without_tag[i]+" "+text_without_tag[i+1] for i in range(0,len(text_without_tag)-1)
            if text_without_tag[i]!='' and text_without_tag[i+1]!='']
    
    counter_text=Counter(text_without_tag)
    counter_bigrams=Counter(bigrams)
    return counter_text,counter_bigrams

# take the most important word of a query (after clean_text function)
# check the associated columns of each word of the query into the tf-idf matrix C
# multiply the tf-idf matrix with the vector of the query
# sort by decreasing order the distances
# return the k higher distances
def get_closest_articles(C,query,k=10):
    a=time.time()
    counter_words,counter_bigrams=get_counters(query)
    keys=[]
    tf_idf=[]
    conn = sqlite3.connect(path+'Database.db')
    c = conn.cursor()
    for counter in [counter_words,counter_bigrams]:
        for key,value in counter.items():
            request=c.execute("Select count(WORD) from Vocabulary where WORD='"+key+"';")
            if c.fetchone()[0]>0:
                request=c.execute("Select M_SUM from Vocabulary where WORD='"+key+"';")
                tf=1+np.log10(value/(len(counter_words)+len(counter_bigrams)))
                idf=np.log10(C.shape[0]/c.fetchone()[0])
                tf_idf.append(tf*idf)
                request=c.execute("Select INDEX_COLUMN from Vocabulary where WORD='"+key+"';")
                keys.append(c.fetchone()[0])
    b=time.time()
    # norm_query=np.linalg.norm(tf_idf)
    # print("norm",norm_query,C[:,keys].shape,np.array(tf_idf).reshape(1,len(tf_idf)).shape)
    numerator = (C[:,keys].multiply(np.array(tf_idf).reshape(1,len(tf_idf)))).sum(axis=1)
    # denominator = np.sqrt(C[:,keys].power(2).sum(axis=1))*norm_query
    # print(numerator,numerator.shape)
    dist = (numerator.reshape(-1))#/denominator.reshape(-1))
    # print(dist)
    argsort_var=np.flip(np.array(np.argsort(dist))[0])
    # lala=[not np.isnan(dist[0,argsort_var[i]]) for i in range(len(argsort_var))]
    # argsort_var=argsort_var[lala]
    # print(argsort_var.shape)
    result_dataset=[]
    conn = sqlite3.connect(path+'Database.db')
    c = conn.cursor()
    for i in range(k):
        # print(argsort_var[i])
        if dist[0,argsort_var[i]]!=0:
            c.execute("Select Title,URL,Location,Line_index,ID_ARTICLE from Articles where ID_ROW="+str(argsort_var[i])+";")
            result_dataset.append(c.fetchone())
    conn.close()
    # print(result_dataset)
    return result_dataset,[dist[0,argsort_var[i]] for i in range(len(result_dataset))]

# takes as input a list of articles ids
# returns the page rank value for each article by giving the id of each article
def get_page_rank(id_article_list):
    conn = sqlite3.connect(path+'Database.db')
    c = conn.cursor()
    dist_pr=[]
    for i in id_article_list:
        c.execute("Select INDEX_ROW from Transitional_id_article_index_R where ID_ARTICLE="+str(i))
        index=c.fetchone()[0]
        c.execute("Select Value from R where ID="+str(index)+";")
        dist_pr.append(float(c.fetchone()[0]))
    conn.close()
    return dist_pr

# takes as input where the wikipedia article is saved (adress), and the corresponding line in the file (each line of the file is an article)
# return a part of the text of the article to show a preview when returning results
def get_text(adress,line,query):
    with open(adress) as f:
        lines = [line.rstrip('\n') for line in f]
    text=json.loads(lines[line])
    text=remove_tag(text['text'])
    text=highlight_text(text,query)
    try:
        start=text.index("<b>")
        text=text[max(0,start-100):start+300]
    except:
        text=text[:400]
    return text

# takes as input the text of a wikipedia article, and the query string
# turn to bold the common word in the query and into the text    
def highlight_text(text,query):
    for i in query:
        text=text.replace(i, "<b>"+i+"</b>")
        text=text.replace(i[0].capitalize()+i[1:], "<b>"+i[0].capitalize()+i[1:]+"</b>")
    return text

# takes as input the w2v vector of each article and the w2v query vector
# return the PCA of the query vector and the w2v vectors of the results
# the query vector is the last element of the array
def get_pca(vector_list,query_vector):
    tmp=np.array(vector_list)[:,:,0]
    tmp=np.concatenate((tmp,query_vector.reshape(1,-1)))
    pca = PCA(n_components=3)
    tmp_transform=pca.fit_transform(tmp)
    return tmp_transform.tolist()[:-1],tmp_transform.tolist()[-1]

# def get_whole_pca(size):
#     try:
#         size=int(size)
#     except:
#         size=1000
#     conn = sqlite3.connect('../Database/word2vec.db')
#     cursor = conn.cursor()
#     request="SELECT * FROM Vectors v WHERE v.ID IN (SELECT ID FROM Vectors ORDER BY RANDOM() LIMIT %d)" % (size)
#     cursor.execute(request)
#     tmp=cursor.fetchall()
#     pca=PCA(3)
#     tmp=pca.fit_transform(np.array(list(tmp))[:,1:])
#     conn.close()
#     return tmp

# takes a query string as input
# scrapes wikipedia pages to get the results articles of this query
# this function is used to compare our results with wikipedia ones
# returns the articles names of the articles
def get_wikipedia_order(query):
    query_clean=query.replace(" ","+")
    query_clean=query_clean.replace("'","%27")
    query_clean=query_clean.replace("é","e")
    query_clean=query_clean.replace("è","e")
    header="https://fr.wikipedia.org/w/index.php?search="
    # add a coma because otherwise wikipedia send me directly on the page
    contents = urllib.request.urlopen(header+query_clean+'+,').read()
    root = ET.fromstring(contents)
    result_title=[]
    for i in root.xpath("//div[@class='mw-search-result-heading']"):
        result_title.append(i.getchildren()[0].values()[1])
    return result_title

# take a word in input
# distorts the word by removing, adding, swaping letters
# returns all possibles words
def modify_word(word):
    letters=list("abcdefghijklmnopqrstuvwxyzéèàêûöù")
    if type(word)==str:
        word_list=list(word)
        
    modify_1_letter=[]
    for i in range(0,len(word_list)):
        for j in range(len(letters)):
            tmp=list(word)
            tmp[i]=letters[j]
            modify_1_letter.append(tmp)

    swap_one_letter=[]
    for i in range(0,len(word_list)-1):
        tmp=list(word)
        a=word_list[i]
        tmp[i]=tmp[i+1]
        tmp[i+1]=a
        swap_one_letter.append(tmp)
    
    insert_one_letter=[]
    for i in range(0,len(word_list)+1):
        for j in range(len(letters)):
            tmp=list(word)
            tmp.insert(i,letters[j])
            insert_one_letter.append(tmp)
    
    missing_one_letter=[]
    for i in range(0,len(word_list)):
        tmp=list(word)
        del tmp[i]
        missing_one_letter.append(tmp)
    
    return [''.join(i) for i in modify_1_letter+swap_one_letter+insert_one_letter+missing_one_letter]

def distance(a,b):
    return np.mean((a.reshape(1,-1)-b.reshape(1,-1))**2)

# takes as input a sentence 
# returns the correct sentence if the original one contains words which don't exists in wikipedia
# the correction if a mix of Peter Norvig article and word2vec distances
def get_correct_sentence(sentence):
    somme=lambda x:sum([i[1] for i in x])
    query_sentence_ori=sentence.lower().replace('\n', ' ').split(' ')
    # query_sentence=sentence.translate(table).lower().replace('\n', ' ').split(' ')
    query_sentence=clean_text(sentence)
    conn = sqlite3.connect(path+'Database.db')
    c = conn.cursor()
    well_spelled_words=[]
    mispelled_words=[]
    
    is_in=lambda x:c.execute("Select count(WORD) from Vocabulary where WORD='"+x+"';").fetchone()[0]>0
    for i in query_sentence:
        if is_in(i):
                well_spelled_words.append(i)
        else:
            mispelled_words.append(i)
    print(well_spelled_words,mispelled_words)
    if len(mispelled_words)==0:
        return ""

    w2v_query=np.mean(np.array([model[i] for i in well_spelled_words if i in model]),axis=0)
    for word in mispelled_words:
        l=[modify_word(word)]+[modify_word(i) for i in set(modify_word(word))]
        reco = set([item for sublist in l for item in sublist if item in voc_dict])
        dict1 = Counter(word) 
        reco_list=[]
        for i in reco:
            dict2 = Counter(i)
            commonDict = dict1 & dict2
            diff1=len(word)/np.sum([value for key,value in commonDict.items()])
            diff2=abs(len(word)-len(i))
            reco_list.append([i,(1/diff1)*(1/(1+diff2))])
        reco_list=sorted(reco_list,key=lambda x:x[1],reverse=True)
        reco_list=[[i[0],float(i[1])/somme(reco_list)] for i in reco_list]
        w2v=[[i[0],distance(model[i[0]],w2v_query)] for i in reco_list if i[0] in model]
        w2v=[[i[0],i[1]/somme(w2v)] for i in w2v]
        both=sorted([[w2v[i][0],-0.8*w2v[i][1]+0.2*reco_list[i][1]] for i in range(len(w2v))],key=lambda x:x[1],reverse=True)
        if len(both)>0:
            query_sentence_ori=[i if i!=word else both[0][0] for i in query_sentence_ori]
            
    conn.close()
    return ' '.join(query_sentence_ori)
    

# take as input a article id
# returns the ids of the wikipedia articles leading to this article id
def get_ids_source_from_id(id_target,nb_nodes):
    conn = sqlite3.connect(path+'Database.db')
    cursor = conn.cursor()
    request="Select ID_source FROM Link_Dict_Inverse WHERE ID_target="+str(id_target)+" LIMIT "+str(nb_nodes)
    cursor.execute(request)
    vertices=[i[0] for i in cursor.fetchall()]
    conn.close()
    return vertices

# takes as input a list of articles ids
# return a list of articles titles and articles urls from a list of articles ids
def get_articles_name_urls_from_ids(ids):
    conn = sqlite3.connect(path+'Database.db')
    cursor = conn.cursor()
    vertices_names=[]
    urls=[]
    for i in ids:
        request="Select Title,URL FROM Articles WHERE ID_ARTICLE='"+str(i)+"'"
        result=cursor.execute(request).fetchone()
        vertices_names.append(result[0])
        urls.append(result[1])
    conn.close()
    return vertices_names,urls

# takes an article id as input
# computes a partial graph of articles leading to this one and their ancestors
# returns 4 lists : list of nodes id, list of articles names/urls, list of edges
def get_graph_from_id(id_graph,nb_nodes=50):
    vertices=get_ids_source_from_id(id_graph,nb_nodes)
    vertices+=[id_graph]
    vertices_names,urls=get_articles_name_urls_from_ids(vertices)
    edges=[(i,id_graph) for i in vertices[:-1]]
    last_vertices=vertices[:-1]
    count=0
    while len(vertices)<nb_nodes:
        nb_ancestors_per_id=int((nb_nodes-len(vertices))/len(vertices))+1
        for i in last_vertices:
            vertices_tmp=get_ids_source_from_id(i,nb_ancestors_per_id)
            vertices_names_tmp,urls_tmp=get_articles_name_urls_from_ids(vertices_tmp)
            edges_tmp=[(j,i) for j in vertices_tmp]
            vertices+=vertices_tmp
            vertices_names+=vertices_names_tmp
            edges+=edges_tmp
            urls+=urls_tmp
            last_vertices=vertices_tmp
        count+=1
        if count==15:
            break
    
    i=0
    while i<len(vertices)-1:
        duplicates=[j for j in range(i+1,len(vertices)) if vertices[j]==vertices[i]]
        vertices=[vertices[i] for i in range(len(vertices)) if i not in duplicates]
        vertices_names=[vertices_names[i] for i in range(len(vertices_names)) if i not in duplicates]
        urls=[urls[i] for i in range(len(urls)) if i not in duplicates]
        i+=1     
    return vertices,vertices_names,edges,urls



