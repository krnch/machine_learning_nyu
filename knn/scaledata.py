import csv

filename = "banknote_train.csv"
filename2="banknote_test.csv"
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    csvreader.next()

    minmax = list()
    colvalues1 = []
    colvalues0= []
    for row in csvreader:
        colvalues0.append(float(row[0]))
        colvalues1.append(float(row[1]))


    minval = min(colvalues0)
    maxval = max(colvalues0)
    minmax.append([minval, maxval])
    minval = min(colvalues1)
    maxval = max(colvalues1)
    minmax.append([minval, maxval])


with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    csvreader.next()
    for row in csvreader:
        firstval= (float(row[0]) - minmax[0][0])/(minmax[0][1] -minmax[0][0])
        secondval= (float(row[1]) - minmax[1][0])/(minmax[1][1] - minmax[1][0])
        thirdval=row[2]

        field =[firstval,secondval,thirdval]
        with open('traindata.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(field)



with open(filename2, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        firstval= (float(row[0]) - minmax[0][0])/(minmax[0][1] -minmax[0][0])
        secondval= (float(row[1]) - minmax[1][0])/(minmax[1][1] - minmax[1][0])
        thirdval=row[2]

        field =[firstval,secondval,thirdval]
        with open('testdata.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(field)


