if bmi<=18.5:
    lst = pd.DataFrame()
    for i in range(len(x)):
      if(x[i]==0):
        lst=lst.append({"Name":onlyveg['Food_items'][i],"Calories":onlyveg['Calories'][i],"Proteins":onlyveg['Proteins'][i],"Fats":onlyveg['Fats'][i]},ignore_index=True)
  
    # return lst
    temp=pd.DataFrame(columns=['Namee','Caloriess','Fats'])
    count=0

    for i in range(len(lst)):
      while(bmr>count):
        temp=temp.append({'Namee':lst['Name'][i],'Caloriess':lst['Calories'][i],'Fats':lst['Fats'][i]},ignore_index=True)
        count=temp['Caloriess'].sum()
        i=i+1
    return temp


elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    for i in range(len(xheal)):
      if(xheal[i]==3):
        lst=lst.append({"Name":onlyveg['Food_items'][i],"Calories":onlyveg['Calories'][i],"Fats":onlyveg['Fats'][i]},ignore_index=True)
  
    temp=pd.DataFrame(columns=['Namee','Caloriess','Fats'])
    count=0

    for i in range(len(lst)):
      while(bmr>count):
        temp=temp.append({'Namee':lst['Name'][i],'Caloriess':lst['Calories'][i],'Fats':lst['Fats'][i]},ignore_index=True)
        count=temp['Caloriess'].sum()
        i=i+1
    return tem