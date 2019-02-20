from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from django.template import RequestContext

import time
import sys

sys.path.append('/home/gabriel/Documents/MPRI/Web_Data_Management/search_engine/search')
from tools import *


# Create your views here.
def home_page(request):
    return render(request,"search.html")


def search(request):
    if 'q' in request.GET:
        query=request.GET['q']
        if query!="":
            dict_response={}
            a=time.time()
            print(query)
            query_clean= clean_text(query)
            b=time.time()
            #result_list=[[title,url,location,line_index]]
            result_list,dist_tf_idf=get_closest_articles(C,query,k=20)
            # print(dist_tf_idf)
            id_list=[i[4] for i in result_list]
            dist_tf_idf=[np.round(i,2) for i in dist_tf_idf]
            titles=[i[0] for i in result_list]
            texts=[get_text(i[2],i[3],query_clean) for i in result_list]#
            urls=[i[1] for i in result_list]
            c=time.time()
            
            # print(id_list)

            dist_pr=get_page_rank(id_list)
            # print(dist_pr)
            d=time.time()
            query_vector=get_vector_word2vec(query_clean)
            
            dist_word2vec,vector_list=get_word2vec_distances(id_list,query_vector)
            e=time.time()

            # wikipedia_order=get_wikipedia_order(query)
            wikipedia_order=[]
            quality_from_wikipedia=[1 if i in wikipedia_order else 0 for i in titles]
            f=time.time()

            query_corrected=get_correct_sentence(query)
            g=time.time()

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
                    pca_articles,pca_query=get_pca(vector_list,query_vector)
                    dict_response["PCA"]=pca_articles
                    dict_response["PCA_QUERY"]=pca_query
    return render_to_response("result.html",dict_response) 



def display_pr_graph(request):
    if 'graph_id' in request.GET:
        try:
            id_graph=int(request.GET['graph_id'])
        except:
            return render_to_response("graph.html",None) 
        vertices,vertices_names,edges,urls=get_graph_from_id(id_graph)
        dict_response={"vertices":zip(vertices,vertices_names,urls),"edges":edges}
        return render_to_response("graph.html",dict_response) 