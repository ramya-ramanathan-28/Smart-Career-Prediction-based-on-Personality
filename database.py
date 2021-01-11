import pickle
import pymysql.cursors
careers=["General Manager","Insurance Agent","Loan Officer","School Administrator","Accountant","Office Manager","Probation Officer","Logistician","Building Contractor","Police detective","Financial Advicer","Sales Manager","Carpenter","Mechanic","Computer Hardware Engineer","Operations Analyst"]
careers2=["Elementary Teacher","Child Care Director","Nutritionist","Cosmetologist","Social Worker","Book Keeper","Medical Secretary","Executive Assistant","Recreation Director","Customer Service Rep","Receptionist","Dental Assistant","Veterenary Technician","Equipment Repairer","Surveyor","Home Health Aide"]
careers3=["Executive","Engineer","Attorney","Architect","Software Develepor","Technical Writer","Judge"," Surgeon","Urban Planner","Entrepreneur","Producer/Director","Real Estate Agent","Software Engineer","Medical Sientist","Mathematician","Psyciatrist"]
careers4=["Non-Profit Director","Teacher","Health Educator","PR Specialist","School Counsellor","Writer","Interior Designer","Pediatrician","Recreational Therapist","Restrauntereur","Pre-school Teacher","Trave; Writer","Animator","Psychologist","Librarian","Author"]
inputFile = 'test2.data'
fd = open(inputFile, 'rb')
b = pickle.load(fd)
b1=pickle.load(fd)
fd.close();
inputFile = 'test.txt'
fd = open(inputFile, 'rb')
arr = pickle.load(fd)
connection = pymysql.connect(host='localhost',
                            user='roshan',
                            password='EDUCATION',
                            db='rohanss6',
                            charset='utf8mb4',
                            cursorclass=pymysql.cursors.DictCursor)
try:
 with connection.cursor() as cursor:
	 sql="delete from ESFJ"
	 cursor.execute(sql)
	 for i in range(0,4):
	     sql="insert into ESFJ values('{}',{})".format(careers2[b1[i]],b[i])
	     cursor.execute(sql)
	 sql="delete from ISFJ"
	 cursor.execute(sql)
	 for i in range(4,8):
	     sql="insert into ISFJ values('{}',{})".format(careers2[b1[i]],b[i])
	     cursor.execute(sql)
	 sql="delete from ESFP"
	 cursor.execute(sql)
	 for i in range(8,12):
	     sql="insert into ESFP values('{}',{})".format(careers2[b1[i]],b[i])
	     cursor.execute(sql)
	 sql="delete from ISFP"
	 cursor.execute(sql)
	 for i in range(12,16):
	     sql="insert into ISFP values('{}',{})".format(careers2[b1[i]],b[i])
	     cursor.execute(sql)
	 for x in arr:
             sql="insert into changed values('{}','{}',{})".format(x[0],x[1],x[2])
             cursor.execute(sql)
 connection.commit() 
finally:
 connection.close()
fw = open(inputFile, 'wb')
pickle.dump(b, fw)
pickle.dump(b1, fw)
fw.close()
outputFile = 'test.txt'
fw = open(outputFile, 'wb')
pickle.dump(arr, fw)
fw.close()
