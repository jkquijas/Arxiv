#Crawl through arxiv.org and create a set of files for each chosen search category
#Programmed by Jonathan Quijas
#Last modified 07/21/2016


import time
from ArxivDataModule import ArxivCrawler

#Clock in time
start = time.time()
#Comment


search_field = 'cat'
n_results = 5
arxiv = ArxivCrawler(n_results, search_field).crawl()
#Clock in time
final_time = time.time() - start
print 'Finished after ', final_time