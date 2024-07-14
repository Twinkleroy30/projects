import csv
import requests
from bs4 import BeautifulSoup

HEADERS = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/90.0.4430.212 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})
  
def getdata(url):
    r = requests.get(url, headers=HEADERS)
    return r.text
  
def mtla(ls1, ls2):
    # create variables to keep track of indices
    i = 0
    j = 0
    # flag to alternate between the lists
    flag = True
    # the resulting list
    res = []
    while i < len(ls1) and j < len(ls2):
        if flag:
            res.append(ls1[i] + '---\n')
            i += 1
            flag = False
        else:
            res.append(ls2[j] + '---\n')
            j += 1
            flag = True
    # append the remaining elements
    if i == len(ls1):
        res = res + ls2[j:]
    else:
        res = res + ls1[i:]
    # return the resulting list
    return res

  
def html_code(url):
  
    # pass the url
    # into getdata function
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
  
    # display html code
    return (soup)

url = input("Enter URL of website:")  
  
  
soup = html_code(url)

def cus_data(soup):
    data_str = ""
    cus_list = []
  
    for item in soup.find_all("span", class_="a-profile-name"):
        data_str = data_str + item.get_text()
        cus_list.append(data_str)
        data_str = "\n"
    return cus_list
  
  
cus_res = cus_data(soup)

def cus_rev(soup):
    data_str = ""
  
    for item in soup.find_all("span", class_="a-size-base review-text"):
        data_str = data_str + item.get_text() 
  
    result = data_str.split("\n")
    return (result)
  
  
rev_data = cus_rev(soup)
rev_result = []
for i in rev_data:
    if i == "":
        pass
    else:
        if i == "Read more":
            pass
        else:
            rev_result.append(i) 


ls1 = cus_res
ls2 = rev_result

# .....:::::  below code is to save results in html format  :::::......

with open('scraper.html', 'w', encoding="utf-8") as f:
    data = rev_result
    for line in data:
        f.write("<br><br><hr><br><i>")
        f.write(line)
        f.write("</i>")
        print('\n' + '_' * 80 + '\n')
        print(line)
    
# .....:::::  below code is to save results in csv format  :::::......
with open('scraper.csv', 'w', encoding="utf-8", newline='') as csvfile:
    fieldnames = ['Sl.No.', 'Review']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    i = 0
    while i < len(rev_result):

        writer.writerow({'Sl.No.': i + 1, 'Review': rev_result[i]})
        i += 1

# .....:::::  below code is to save results in html format  :::::......
print('\n'  + '_' * 40 + 'END of REPORT' + '_' * 40 +'\n')
