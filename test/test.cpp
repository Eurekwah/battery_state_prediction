#include <iostream>
#include <string>

using namespace std;
void mystrrev(char *str)   //(char string[30]) 传参要传指针或者引用，比如我给别人指路去你家，我会告诉别人门牌号，不会直接把你家拆了搬过去
{
	int i,c;
	for(i=0;i<30;i++)
	{
		if(str[i] == '\0')//(string[i]==32) 字符串的结尾是\0 不是空格，不能等于32
		{
			c=i-1;
            break;
		}
	} 
	for(i=c;i>=0;i--)
	{
		cout<<str[i];
	}
}
int main()
{

	char string[30];
	int i;
	/*for(i=0;i<30;i++)
	{
		cin>>string[i];
		if(string[i]==32)
		{
			break;
		}
	 }
	 mystrrev(str);*/
	 char str[6]="abcde";
     if(str[5] == '\0')
	 mystrrev(str);
	 return 0;
}

/*
简单写法
void strbkwd(string &a)
{
    string c;
    copy(a.rbegin(), a.rend(), back_inserter(c));
    cout << c;
}

int main()
{
    string a;
    cin >> a;
    strbkwd(a);
    return 0;
}
*/

