def labels(path):
    file = open(path, mode='r')
    initialList = file.readlines()
    finalList =[]
    for l in initialList:
        finalList.append(l.replace("\n",""))
    return finalList