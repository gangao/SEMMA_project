libname se 'D:\Analysis\SEMMA_project';
/*data se.df_undp;*/
/*set df_undp;*/
/*run;*/

%let varlist =var1 var2 var3;

proc logistic data=se.df_undp;
	model perf(event='1')= &varlist. /selection=stepwise details maxstep=4 sle=0.2 sls=0.2;
	weight wt; 
run;
/*Warning: The maximum likelihood estimate does not exist.*/
/*Warning: The maximum likelihood estimate does not exist.*/

libname se 'D:\Analysis\SEMMA_project';
/*data se.anes96;*/
/*	set anes96;*/
/*run;*/

%let varlist = logpopul TVnews selfLR ClinLR age educ income;
proc logistic data=se.anes96;
	model vote(event='1')= &varlist. /selection=stepwise details maxstep=2;
run;

/*data se.spector_data;*/
/*	set spector_data;*/
/*run;*/

%let varlist = GPA TUCE PSI;
%let indata=se.spector_data;
%let weight=wt;
%let target=GRADE;

proc logistic data=&indata.;
	model &target.(event='1')= &varlist. /selection=stepwise details maxstep=1;
	weight &weight.;
run;



/*1 - compare vif - no weight*/
ods output parameterestimates=reg;

proc reg data=&indata.;
	model &target.=&varlist. /vif;
run;

/*参数估计 */
/*变量 自由度 参数估计 标准误差 t值 Pr>|t| 方差膨胀 */
/*Intercept 1 -1.49802 0.52389 -2.86 0.0079 0 */
/*GPA 1 0.46385 0.16196 2.86 0.0078 1.17616 */
/*TUCE 1 0.01050 0.01948 0.54 0.5944 1.18944 */
/*PSI 1 0.37855 0.13917 2.72 0.0111 1.01290 */
1.1761582341579837
1.1894350280708073
1.012902241028604

proc reg data=&indata.;
	model GPA= TUCE PSI;
run;

/**/
/*均方根误差 0.44494 R 方 0.1498 */
/*因变量均值 3.11719 调整 R 方 0.0911 */
/*变异系数 14.27367     */
/*TOL = 1-0.1498 =0.8502*/
/*VIF = 1/TOL = 1.176193‬*/
/*TOL_ADJ = 1-0.0911 =0.9089*/
/*VIF_ADJ = 1/TOL_ADJ = 1.100231‬*/

/*using no adjust R square to calc vif */

/*2 - compare vif - with weight*/
proc reg data=&indata.;
	model &target.=&varlist. /vif;
	weight &weight.;

run;

/*参数估计 */
/*变量 自由度 参数估计 标准误差 t值 Pr>|t| 方差膨胀 */
/*Intercept 1 -1.03932 0.51610 -2.01 0.0537 0 */
/*GPA 1 0.41306 0.14697 2.81 0.0089 1.10718 */
/*TUCE 1 -0.00506 0.01938 -0.26 0.7959 1.17791 */
/*PSI 1 0.48714 0.13891 3.51 0.0015 1.07917 */



proc reg data=&indata.;
	model GPA= TUCE PSI;
	weight &weight.;
run;



/*均方根误差 1.05506 R 方 0.0968 */
/*因变量均值 3.07282 调整 R 方 0.0345 */
/*变异系数 34.33510     */
/*TOL = 1-0.0968 =0.9032‬*/
/*VIF = 1/TOL = 1.107174490‬*/

/*still using no adjust R square to calc vif */
