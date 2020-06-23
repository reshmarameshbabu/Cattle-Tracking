rows = open("urls.txt").read().strip().split("\n")
rows1 = open("urls1.txt").read().strip().split("\n")
total = 0
print("urls",len(rows))
print("urls",len(rows1))
l = []
# loop the URLs
for url in rows1:
    if url not in rows:
        l.append(url)
print(len(l))
#with open('new_urls.txt', 'w') as f:
#    for item in l:
#        f.write("%s\n" % item)
