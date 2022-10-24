#include<stdio.h>
int stack1[100];
int stack2[100];
int top=-1,size;
void push_s1()
{
    if(top==size-1)
    printf("Overflow\n");
    else{
        int x;
        printf("\nEnter the value ");
        scanf("%d",&x);
        stack1[++top]=x;
    }
}


void push_s2(){
    for(int i=0,j=top;j>=0;j--,i++)
    stack2[j]=stack1[i];
}


void push_qstack(){
    for(int i=0,j=top-1;i<=top-1;j--,i++)
    stack1[j]=stack2[i];
    top--;
}


void pop_queue(){
    if(top==-1)
    printf("underflow");
    else{
        push_s2();
        printf("Value popped is %d",stack2[top]);
        push_qstack();
    }
}

int main()
{
    printf("Enter the size ");
    scanf("%d",&size);
    // size=x;
    int ch;
    for(int i=0;i!=1;){
        printf("\n1 for push 2 for pop 3 for exit ");
        scanf("%d",&ch);
        switch(ch){
            case 1:
            push_s1();
            break;
            case 2:
            pop_queue();
            break;
            case 3:
            i=1;
            break;
            default:
            printf("Enter the right choice");
            
        }
    }
}
