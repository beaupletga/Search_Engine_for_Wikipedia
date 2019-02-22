from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from django.template import RequestContext

import time
import sys

sys.path.append('/home/gabriel/Documents/MPRI/Web_Data_Management/search_engine/search')
from tools import *


# return the home page
def home_page(request):
    return render(request,"search.html")

# main function
# this function is trigerred when the user makes a query
def search(request):
    if 'q' in request.GET:
        query=request.GET['q']
        try:
            k=int(request.GET['k'])
        except:
            k=20
        if query!="":
            dict_response={}
            a=time.time()
            print(query)
            # if the first and the last character are ' then we keep all words
            if query[0]==query[-1]=="'":
                query_clean=query[1:-1].split(" ")
            else:
                query_clean= clean_text(query)
            print(query_clean)
            b=time.time()
            # returns the closest articles and theirs distances based on the tf-idf matrix
            result_list,dist_tf_idf=get_closest_articles(C,query,k=k)
            id_list=[i[4] for i in result_list]
            titles=[i[0] for i in result_list]
            texts=[get_text(i[2],i[3],query_clean) for i in result_list]#
            urls=[i[1] for i in result_list]
            c=time.time()
            
            # print(id_list)
            # get the page rank value of each article
            dist_pr=get_page_rank(id_list)
            # print(dist_pr)
            d=time.time()
            # get the word2vec vector of each article
            query_vector=get_vector_word2vec(query_clean)
            # computes the distances between the w2v vector of the query and the article ones
            dist_word2vec,vector_list=get_word2vec_distances(id_list,query_vector)
            e=time.time()

            # get the names of the articles for the same query in wikipedia
            # wikipedia_order=get_wikipedia_order(query)
            wikipedia_order=[]
            quality_from_wikipedia=[1 if i in wikipedia_order else 0 for i in titles]
            f=time.time()

            # correct the sentence if necessary
            # we put the function at the end, because the correction is only proposed and not forced
            query_corrected=get_correct_sentence(query)
            g=time.time()

            dist_tf_idf=[round(float(i)/sum(dist_tf_idf),4) for i in dist_tf_idf]
            dist_pr=[round(float(i)/sum(dist_pr),7) for i in dist_pr]
            dist_word2vec=[round(float(i)/sum(dist_word2vec),4) for i in dist_word2vec]

            dict_response={'zip_data':zip(id_list,titles,texts,urls,dist_tf_idf,dist_pr,dist_word2vec,quality_from_wikipedia),
											'time_clean_text':round(b-a,3),
                                            'Last_query':query,
                                            'titles':titles,
											'time_tf_idf':round(c-b,3),
											'time_w2v':round(e-d,3),
                                            'time_wikipedia':round(f-e,3),
                                            'time_correcting':round(g-f,3),
                                            'PCA':[],
                                            'PCA_QUERY':[],
                                            'query_corrected':query_corrected}
            if 'PCA' in request.GET:
                if request.GET['PCA']=="checked":
                    # returns also the PCA
                    pca_articles,pca_query=get_pca(vector_list,query_vector)
                    dict_response["PCA"]=pca_articles
                    dict_response["PCA_QUERY"]=pca_query
                    print(pca_query)
            return render_to_response("result.html",dict_response) 
    # when nothing is queried, a gif of Hal appears
    return render_to_response("result_hal.html",None) 

# displays the graph of the ancestors of an article
def display_pr_graph(request):
    if 'graph_id' in request.GET:
        try:
            id_graph=int(request.GET['graph_id'])
        except:
            return render_to_response("graph.html",None)
        if 'nb_nodes' in request.GET:
            try:
                nb_nodes=int(request.GET['nb_nodes'])
            except:
                nb_nodes=50

        vertices,vertices_names,edges,urls=get_graph_from_id(id_graph,nb_nodes=nb_nodes)
        dict_response={"vertices":zip(vertices,vertices_names,urls),"edges":edges}
        return render_to_response("graph.html",dict_response) 