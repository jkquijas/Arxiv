from phpserialize import unserialize

filePath = '/home/jonathan/Desktop/ACMSerializedCoPPubs/serializedPubsCS.txt'
data = raw_data = open(filePath, 'r').read()
data = unserialize(data)
numPapers = len(data)
print data[0].keys()
#['ccsData', 'pubAbstract', 'pubURL', 'pubConcepts', 'pubTitle', 'pubAuthorTags', 'relAuthors']
for i in range(numPapers):
    if(len(data[i]['pubConcepts']) == 0):
        continue
    print data[i]['pubAbstract']
    for j in range(len(data[i]['pubConcepts'])):
        print data[i]['pubConcepts'][j][0], " "
    print "++++++++++++++++"

    print "-----------------------------------------------------------"""
