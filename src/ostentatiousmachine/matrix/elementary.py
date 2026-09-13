import torch


def rowswap(matrix,source_row,target_row):
    result=matrix.clone()
    temp=result[source_row].clone()
    result[source_row]=result[target_row]
    result[target_row]=temp

    return result

def rowscale(matrix,source_row,scaling_factor):
    result=matrix.clone()
    result[source_row]=result[source_row]*scaling_factor

    return result

def rowreplacement(matrix,first_row,second_row,j,k):
    result=matrix.clone()
    result[second_row]=(j*result[first_row]+k*result[second_row])

    return result

def rref(matrix):
    result=matrix.clone().float()
    rows,cols=result.shape
    pivot_row=0

    for col in range(cols):
        if pivot_row>=rows:
            break

        swap_row=None

        for row in range(pivot_row,rows):
            if not torch.isclose(result[row,col],torch.tensor(0.0)):
                swap_row=row
                break

        if swap_row is None:
            continue

        if swap_row!=pivot_row:
            result=rowswap(result,pivot_row,swap_row)

        pivot_value=result[pivot_row,col]
        result=rowscale(result,pivot_row,1.0/pivot_value)

        for row in range(rows):
            if row!=pivot_row:
                factor=-result[row,col]
                result[row]=(result[row]+factor*result[pivot_row])

        pivot_row+=1

    return result


if __name__=="__main__":

    matrix=torch.tensor([[1.,3.,0.,0.,3.],[0.,0.,1.,0.,9.],[0.,0.,0.,1.,-4.]])

    print("original Matrix:")
    print(matrix)

    print("\nrow swap R1 <-> R2:")
    swapped=rowswap(matrix,0,1)
    print(swapped)

    print("\nscale R1 by 1/3:")
    scaled=rowscale(swapped,0,1/3)
    print(scaled)

    print("\nreplace R3=-3R1+R3:")
    replaced=rowreplacement(scaled,0,2,-3,1)
    print(replaced)

    print("\nRREF:")
    print(rref(matrix))