#include <stdio.h>
#include<math.h>
int top=-1;
int pop(int arr[]){
    if(top==-1)
    printf("Stack underfow\n");
    else
    return (arr[top--]);
}
void push(int arr[],int a){
    if(a==1){
        int val;
        printf("Enter value ");
        scanf(" %d",&val);
        arr[++top]=val;
    }
    else
    {
        char ch,c,a,b;
        a=pop(arr);
        b=pop(arr);
        printf("Enter the operator ");
        scanf(" %c", &ch);
        switch(ch){
            case '+':
            c=b+a;
            break;
            case '-':
            c=b-a;
            break;
            case '*':
            c=b*a;
            break;
            case '%':
            c=b%a;
            break;
            case '/':
            c=b/a;
            break;
            case '^':
            c=pow(b,a);
            break;
        }
        arr[++top]=c;
    }
}
void display(int arr[]){
    printf("Output is %d \n",arr[top]);
}
void main(){
    int loop=0;
    int arr[50];
    while(loop!=1){
        printf("1 for value\t2 for operator\t3 for display\t4 for exit \n");
        int choices;
        scanf("%d",&choices);
        switch(choices){
            case 1:
            push(arr,choices);
            break;
            case 2:
            push(arr,choices);
            break;
            case 3:
            display(arr);
            break;
            case 4:
            loop=1;
            break;
            
            default:
            printf("\nEnter the right choices\n");
        }
    }
}
