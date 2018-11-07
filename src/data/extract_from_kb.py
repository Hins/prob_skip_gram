# -*- coding: utf-8 -*-
# @Time        : 2018/9/7 10:49
# @Author      : panxiaotong
# @Description :  extract entities out of Google Knowledge base

'''
import json
import requests
from wikidata.client import Client

WikiData_url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&language=en&format=json&search="
request_header = {'accept':'application/json'}
r = requests.get(WikiData_url + "lion", headers=request_header)
if r.status_code == 200:
    json_obj = json.loads(str(r.text))
    print(json_obj["search"][0]["id"])
    client = Client()
    entity = client.get(json_obj["search"][0]["id"], load=True)
    for k,v in entity.data.items():
        print("%s : %s" % (k,v))
    for k, v in entity.attributes.items():
        print("%s : %s" % (k, v))
    for k,v in entity.label.texts.items():
        print("%s : %s" % (k, v))
    for k,v in entity.description.texts.items():
        print("%s : %s" % (k, v))
    for k,v in entity.client.identity_map.items():
        print("%s : %s" % (k, v))
    print(entity.type.)
    #print(type(entity.attributes.items()['claims']))
'''


import urllib
import json
import requests
import sys

class GooleKGAPI(object):
    def __init__(self):
        self.api_key = 'AIzaSyAvXCjcZCh7QRMAgcppheJkfUWktGZQg_M'

    def getResult(self,query):
        params = {
            'query': query,
            'limit': 1,
            'indent': True,
            'key': self.api_key,
        }
        service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
        url = service_url + '?' + urllib.urlencode(params)
        proxies = {
            "http": "http://106.46.136.112:808",
            "https": "http://106.46.136.112:808",
        }
        response = json.loads(requests.get(url=url, proxies=proxies).text)
        if "itemListElement" not in response or len(response['itemListElement']) == 0 or \
            "result" not in response['itemListElement'][0] or "@type" not in response['itemListElement'][0]['result']:
            return ""
        return response['itemListElement'][0]['result']['@type']

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("<extract_from_kb> <id file> <word dict file> <kb output> <kb dict output>")
        sys.exit()

    gkg = GooleKGAPI()
    word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[int(elements[1])] = elements[0]
        f.close()

    kb_output = open(sys.argv[3], 'w')
    kb_dict = {}
    kb_cache_dict = {}
    kb_dict_output = open(sys.argv[4], 'w')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            id = int(line.strip('\r\n'))
            if id not in word_dict:
                kb_output.write("0\n")
                kb_output.flush()
            else:
                if word_dict[id] not in kb_cache_dict:
                    kb_list = gkg.getResult(word_dict[id])
                    if isinstance(kb_list, str):
                        kb_output.write("0\n")
                        kb_output.flush()
                        kb_cache_dict[word_dict[id]] = "0"
                        continue
                    kb_list_id = []
                    for entity in kb_list:
                        if entity not in kb_dict:
                            kb_dict[entity] = len(kb_dict) + 1
                        kb_list_id.append(kb_dict[entity])
                    kb_cache_dict[word_dict[id]] = ",".join([str(item) for item in kb_list_id])
                    kb_output.write(",".join([str(item) for item in kb_list_id]) + "\n")
                    kb_output.flush()
                else:
                    kb_output.write(kb_cache_dict[word_dict[id]] + "\n")
                    kb_output.flush()
        f.close()
    kb_output.close()

    for k,v in kb_dict.items():
        kb_dict_output.write(k + "\t" + str(v) + "\n")
    kb_dict_output.close()
