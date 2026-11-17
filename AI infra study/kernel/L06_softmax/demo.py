from softmax import reference_rows, simulate_padded
x=[[1000.,1001.,1002.],[-2.,-2.,-2.]]
for row in x:
    y,b=simulate_padded(row)
    print('row:',row,'BLOCK:',b,'probabilities:',[round(v,6) for v in y],'sum:',round(sum(y),6))
