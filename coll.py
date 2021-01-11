import itertools as it
import pickle
careers=["General Manager","Insurance Agent","Loan Officer","School Administrator","Accountant","Office Manager","Probation Officer","Logistician","Building Contractor","Police detective","Financial Advicer","Sales Manager","Carpenter","Mechanic","Computer Hardware Engineer","Operations Analyst"]
careers2=["Elementary Teacher","Child Care Director","Nutritionist","Cosmetologist","Social Worker","Book Keeper","Medical Secretary","Executive Assistant","Recreation Director","Customer Service Rep","Receptionist","Dental Assistant","Veterenary Technician","Equipment Repairer","Surveyor","Home Health Aide"]
careers3=["Executive","Engineer","Attorney","Architect","Software Develepor","Technical Writer","Judge"," Surgeon","Urban Planner","Entrepreneur","Producer/Director","Real Estate Agent","Software Engineer","Medical Sientist","Mathematician","Psyciatrist"]
careers4=["Non-Profit Director","Teacher","Health Educator","PR Specialist","School Counsellor","Writer","Interior Designer","Pediatrician","Recreational Therapist","Restrauntereur","Pre-school Teacher","Trave; Writer","Animator","Psychologist","Librarian","Author"]
def careerRet(prs):
        inputFile = 'test.data'
        fd = open(inputFile, 'rb')
        a = pickle.load(fd)
        a1=pickle.load(fd)
        inputFile = 'test2.data'
        fd = open(inputFile, 'rb')
        b = pickle.load(fd)
        b1=pickle.load(fd)
        inputFile = 'test3.data'
        fd = open(inputFile, 'rb')
        c = pickle.load(fd)
        c1=pickle.load(fd)
        inputFile = 'test4.data'
        fd = open(inputFile, 'rb')
        d = pickle.load(fd)
        d1=pickle.load(fd)
        res=[]
        if prs=="ESTJ":
            r=0
            for i in range(0,4):
                if a[i]>=7:
                    r=r+1
                    res = res + [(careers[a1[i]], a[i]*10)]
        elif prs=="ISTJ":
            r=0
            print ('in here')
            for i in range(4,8):
                if a[i]>=7:
                    r=r+1
                    print (careers[a1[i]], a[i]*10)
                    res = res + [(careers[a1[i]], a[i]*10)]
           
        elif prs=="ESTP":
            r=0
            for i in range(8,12):
                if a[i]>=7:
                    r=r+1
                    res = res + [(careers[a1[i]], a[i]*10)] 
            
        elif prs=="ISTP":
            r=0
            for i in range(12,16):
                if a[i]>=7:
                    r=r+1
                    res = res + [(careers[a1[i]], a[i]*10)] 
           
        elif prs=="ESFJ":
            r=0
            for i in range(0,4):
                if b[i]>=7:
                    r=r+1
                    res = res + [(careers2[b1[i]], b[i]*10)] 
        elif prs=="ISFJ":
            r=0
            for i in range(4,8):
                if a[i]>=7:
                    r=r+1
                    res = res + [(careers2[b1[i]], b[i]*10)] 
        elif prs=="ESFP":
            r=0
            for i in range(8,12):
                if b[i]>=7:
                    r=r+1
                    res = res + [(careers2[b1[i]], b[i]*10)] 
           
        elif prs=="ISFP":
            r=0
            for i in range(12,16):
                if b[i]>=7:
                    r=r+1
                    res = res + [(careers2[b1[i]], b[i]*10)] 
        elif prs=="ENTJ":
            r=0
            for i in range(0,4):
                if c[i]>=7:
                    r=r+1
                    res = res + [(careers3[c1[i]], c[i]*10)] 
        elif prs=="INTJ":
            r=0
            
            for i in range(4,8):
                if c[i]>=7:
                    r=r+1
                    res = res + [(careers3[c1[i]], c[i]*10)]
        elif prs=="ENTP":
            r=0
            
            for i in range(8,12):
                if c[i]>=7:
                    r=r+1
                    res = res + [(careers3[c1[i]], c[i]*10)]
        elif prs=="INTP":
            r=0
            for i in range(12,16):
                if c[i]>=7:
                    r=r+1
                    res = res + [(careers3[c1[i]], c[i]*10)]
        elif prs=="ENFJ":
            r=0
            for i in range(0,4):
                if d[i]>=7:
                    r=r+1
                    res = res + [(careers4[d1[i]], d[i]*10)]
        elif prs=="INFJ":
            r=0
            for i in range(4,8):
                if d[i]>=7:
                    r=r+1
                    res = res + [(careers4[d1[i]], d[i]*10)]
        elif prs=="ENFP":
            r=0
            for i in range(8,12):
                if d[i]>=7:
                    r=r+1
                    res = res + [(careers4[d1[i]], d[i]*10)]
        elif prs=="INFP":
            r=0
            for i in range(12,16):
                if d[i]>=7:
                    r=r+1
                    res = res + [(careers4[d1[i]], d[i]*10)]
        outputFile = 'test.data'
        fw = open(outputFile, 'wb')
        pickle.dump(a, fw, protocol = 2)
        pickle.dump(a1, fw, protocol = 2)
        fw.close()
        outputFile = 'test2.data'
        fw = open(outputFile, 'wb')
        pickle.dump(b, fw, protocol = 2)
        pickle.dump(b1, fw, protocol = 2)
        fw.close()
        outputFile = 'test3.data'
        fw = open(outputFile, 'wb')
        pickle.dump(c, fw, protocol = 2)
        pickle.dump(c1, fw, protocol = 2)
        fw.close()
        outputFile = 'test4.data'
        fw = open(outputFile, 'wb')
        pickle.dump(d, fw, protocol = 2)
        pickle.dump(d1, fw, protocol = 2)
        fw.close()

        return res
                                
def career(prs,fb):
        inputFile = 'test.data'
        fd = open(inputFile, 'rb')
        a = pickle.load(fd)
        a1=pickle.load(fd)
        inputFile = 'test2.data'
        fd = open(inputFile, 'rb')
        b = pickle.load(fd)
        b1=pickle.load(fd)
        inputFile = 'test3.data'
        fd = open(inputFile, 'rb')
        c = pickle.load(fd)
        c1=pickle.load(fd)
        inputFile = 'test4.data'
        fd = open(inputFile, 'rb')
        d = pickle.load(fd)
        d1=pickle.load(fd)
        res=[]
        if prs=="ESTJ":
            f=-1
            for i in range(0,4):
                f=f+1
                x=fb[f]
                #print(careers[a1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        a[i]=a[i]*0.9+x*0.1
                        if a[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(4,16)):
                                if (a[l]>max) and (a1[l] not in (a1[0],a1[1],a1[2],a1[3])) :
                                        max=a[l]
                                        pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ESTJ",careers[a1[i]],a[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            a[i]=a[pos]
                            a1[i]=a1[pos]                        
        if prs=="ISTJ":
            f=-1  
            for i in range(4,8):
                f=f+1
                x=fb[f]
                #print(careers[a1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        a[i]=a[i]*0.9+x*0.1
                        if a[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,4),range(8,16)):
                                if (a[l]>max) and (a1[l] not in (a1[4],a1[5],a1[6],a1[7])):
                                    max=a[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ISTJ",careers[a1[i]],a[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            a[i]=a[pos]
                            a1[i]=a1[pos]                       
        if prs=="ESTP":
            f=-1    
            for i in range(8,12):
                f=f+1
                x=fb[f]
                #print(careers[a1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        a[i]=a[i]*0.9+x*0.1
                        if a[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,8),range(12,16)):
                                if (a[l]>max) and (a1[l] not in (a1[8],a1[9],a1[10],a1[11])):
                                    max=a[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ESTP",careers[a1[i]],a[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            a[i]=a[pos]
                            a1[i]=a1[pos]   
        if prs=="ISTP":
            f=-1     
            for i in range(12,16):
                f=f+1
                x=fb[f]
                #print(careers[a1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        a[i]=a[i]*0.9+x*0.1
                        if a[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,12)):
                                if (a[l]>max) and (a1[l] not in (a1[12],a1[13],a1[14],a1[15])):
                                    max=a[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ISTP",careers[a1[i]],a[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            a[i]=a[pos]
                            a1[i]=a1[pos]   
        if prs=="ESFJ":
            f=-1  
            for i in range(0,4):
                f=f+1
                x=fb[f]
                #print(careers2[b1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        b[i]=b[i]*0.9+x*0.1
                        if b[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(4,16)):
                                if (b[l]>max) and (b1[l] not in (b1[0],b1[1],b1[2],b1[3])) :
                                        max=b[l]
                                        pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ESTJ",careers2[b1[i]],b[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            b[i]=b[pos]
                            b1[i]=b1[pos]                        
        if prs=="ISFJ":
            f=-1      
            for i in range(4,8):
                f=f+1
                x=fb[f]
                #print(careers2[b1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        b[i]=b[i]*0.9+x*0.1
                        if b[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,4),range(8,16)):
                                if (b[l]>max) and (b1[l] not in (b1[4],b1[5],b1[6],b1[7])):
                                    max=b[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ISFJ",careers2[b1[i]],b[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            b[i]=b[pos]
                            b1[i]=b1[pos]                       
        if prs=="ESFP":
            f=-1
            for i in range(8,12):
                f=f+1
                x=fb[f]
                #print(careers2[b1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        b[i]=b[i]*0.9+x*0.1
                        if b[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,8),range(12,16)):
                                if (b[l]>max) and (b1[l] not in (b1[8],b1[9],b1[10],b1[11])):
                                    max=b[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ESFP",careers2[b1[i]],b[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            b[i]=b[pos]
                            b1[i]=b1[pos]   
        if prs=="ISFP":
            f=-1
            for i in range(12,16):
                f=f+1
                x=fb[f]
                #print(careers2[b1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        b[i]=b[i]*0.9+x*0.1
                        if b[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,12)):
                                if (b[l]>max) and (b1[l] not in (b1[12],b1[13],b1[14],b1[15])):
                                    max=b[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ISFP",careers2[b1[i]],b[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            b[i]=b[pos]
                            b1[i]=b1[pos]   
        if prs=="ENTJ":
            f=-1
            for i in range(0,4):
                f=f+1
                x=fb[f]
                #print(careers3[c1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        c[i]=c[i]*0.9+x*0.1
                        if c[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(4,16)):
                                if (c[l]>max) and (c1[l] not in (c1[0],c1[1],c1[2],c1[3])) :
                                        max=c[l]
                                        pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ESTJ",careers3[c1[i]],c[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            c[i]=c[pos]
                            c1[i]=c1[pos]                        
        if prs=="INTJ":
            f=-1
            for i in range(4,8):
                f=f+1
                x=fb[f]
                #print(careers3[c1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        c[i]=c[i]*0.9+x*0.1
                        if c[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,4),range(8,16)):
                                if (c[l]>max) and (c1[l] not in (c1[4],c1[5],c1[6],c1[7])):
                                    max=c[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["INTJ",careers3[c1[i]],c[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            c[i]=c[pos]
                            c1[i]=c1[pos]                       
        if prs=="ENTP":
            f=-1      
            for i in range(8,12):
                f=f+1
                x=fb[f]
                #print(careers3[c1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        c[i]=c[i]*0.9+x*0.1
                        if c[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,8),range(12,16)):
                                if (c[l]>max) and (c1[l] not in (c1[8],c1[9],c1[10],c1[11])):
                                    max=c[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ENTP",careers3[c1[i]],c[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            c[i]=c[pos]
                            c1[i]=c1[pos]   
        if prs=="INTP":
            f=-1
            for i in range(12,16):
                f=f+1
                x=fb[f]
                #print(careers3[c1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        c[i]=c[i]*0.9+x*0.1
                        if c[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,12)):
                                if (c[l]>max) and (c1[l] not in (c1[12],c1[13],c1[14],c1[15])):
                                    max=c[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ISTP",careers3[c1[i]],c[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            c[i]=c[pos]
                            c1[i]=c1[pos]   
        if prs=="ENFJ":
            f=-1
            for i in range(0,4):
                f=f+1
                x=fb[f]
                #print(careers4[d1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        d[i]=d[i]*0.9+x*0.1
                        if d[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(4,16)):
                                if (d[l]>max) and (d1[l] not in (d1[0],d1[1],d1[2],d1[3])) :
                                        max=d[l]
                                        pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ESTJ",careers4[d1[i]],d[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            d[i]=d[pos]
                            d1[i]=d1[pos]                        
        if prs=="INFJ":
            f=-1      
            for i in range(4,8):
                f=f+1
                x=fb[f]
                #print(careers4[d1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        d[i]=d[i]*0.9+x*0.1
                        if d[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,4),range(8,16)):
                                if (d[l]>max) and (d1[l] not in (d1[4],d1[5],d1[6],d1[7])):
                                    max=d[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["INFJ",careers4[d1[i]],d[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            d[i]=d[pos]
                            d1[i]=d1[pos]                       
        if prs=="ENFP":
            f=-1
            for i in range(8,12):
                f=f+1
                x=fb[f]
                #print(careers4[d1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        d[i]=d[i]*0.9+x*0.1
                        if d[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,8),range(12,16)):
                                if (d[l]>max) and (d1[l] not in (d1[8],d1[9],d1[10],d1[11])):
                                    max=d[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ENFP",careers4[d1[i]],d[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            d[i]=d[pos]
                            d1[i]=d1[pos]   
        if prs=="INFP":
            f=-1
            for i in range(12,16):
                f=f+1
                x=fb[f]
                #print(careers4[d1[i]])
                #x=int(input("Enter your value: "))
                if x>0:
                    if x<=10:
                        d[i]=d[i]*0.9+x*0.1
                        if d[i]<7:
                            max=0
                            pos=0
                            for l in it.chain(range(0,12)):
                                if (d[l]>max) and (d1[l] not in (d1[12],d1[13],d1[14],d1[15])):
                                    max=d[l]
                                    pos=l
                            inputFile = 'test.txt'
                            fd = open(inputFile, 'rb')
                            arr = pickle.load(fd)
                            temp=["ISFP",careers4[d1[i]],d[i]]
                            arr.append(temp)
                            #print(arr)
                            outputFile = 'test.txt'
                            fw = open(outputFile, 'wb')
                            pickle.dump(arr, fw)
                            fw.close()
                            d[i]=d[pos]
                            d1[i]=d1[pos]   
        #print(a)
        #print(a1) 
        #print(b)
        #print(b1)
        #print(c)
        #print(c1)
        #print(d)
        #print(d1)   
        outputFile = 'test.data'
        fw = open(outputFile, 'wb')
        pickle.dump(a, fw, protocol = 2)
        pickle.dump(a1, fw, protocol = 2)
        fw.close()
        outputFile = 'test2.data'
        fw = open(outputFile, 'wb')
        pickle.dump(b, fw, protocol = 2)
        pickle.dump(b1, fw, protocol = 2)
        fw.close()
        outputFile = 'test3.data'
        fw = open(outputFile, 'wb')
        pickle.dump(c, fw, protocol = 2)
        pickle.dump(c1, fw, protocol = 2)
        fw.close()
        outputFile = 'test4.data'
        fw = open(outputFile, 'wb')
        pickle.dump(d, fw, protocol = 2)
        pickle.dump(d1, fw, protocol = 2)
        fw.close()

#career("INFJ")
'''
if prs=="ISTJ":
    print ("The careers that would suit you the most are:")
    for i in a2:
        print(careers[i])
    print("How much in agreement are you that this career would suit you on a scale of 10?")
    j=4        
    for i in a2:
        print(careers[i])
        x=int(input("Enter your value: "))
        if x>0:
            if x<=10:
                a[j]=a[j]*0.9+x*0.1
                j=j+1
'''

