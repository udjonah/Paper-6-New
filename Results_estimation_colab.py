# Estimation code for Section 3 (Results). Runs in Google Colab; seeds fixed for reproducibility.
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Probit
from statsmodels.stats.outliers_influence import variance_inflation_factor
# In Google Colab: !pip install linearmodels   (statsmodels is preinstalled)
DATA='WBES_AI_analysis_clean.xlsx'   # upload the cleaned workbook or point to Google Drive
df=pd.read_excel(DATA,sheet_name='Analysis_Data')
d=df[df.sme==1].copy()
d['cs']=d.country.astype(str)+'_'+d.sector_a4a.astype(int).astype(str)
d['cl']=d.country.astype(str)+'_'+d.strata_baseline.astype(int).astype(str)
CTRL=['ln_emp','ln_age','exporter','foreign_own','mgr_exp','quality_cert','informal_comp','website','sales_growth_ann_w']
X=' + '.join(CTRL)
base=['ai_adopt','ai_depth','cc_full','w0','cs','cl']+CTRL
d['w0']=d.wt_ai
a=d.dropna(subset=[c for c in base if c!='w0']).copy(); a['w']=a.wt_ai/a.wt_ai.mean()
R={'N_main':len(a),'clusters':a.cl.nunique()}
def probit(f,data):
    data=data.copy(); data['w']=data.wt_ai/data.wt_ai.mean()
    return smf.glm(f,data,family=Binomial(link=Probit()),var_weights=data.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(data.cl)[0]})
from scipy.stats import norm
def _X(res,var,val):
    names=list(res.model.exog_names); X=res.model.exog.copy()
    j=names.index(var); X[:,j]=val
    for k,n in enumerate(names):
        if n.startswith(var+':'):
            other=n.split(':',1)[1]; X[:,k]=val*res.model.exog[:,names.index(other)] if other in names else val
    return X
def ame(res,data=None,var='cc_full',mask=None):
    b=res.params.values; w=res.model.var_weights if res.model.var_weights is not None else np.ones(res.model.exog.shape[0])
    X1=_X(res,var,1); X0=_X(res,var,0); dd=norm.cdf(X1@b)-norm.cdf(X0@b)
    if mask is not None: dd=dd[mask]; w=w[mask]
    return float(np.average(dd,weights=w))
def ame_se(res,data=None,var='cc_full',reps=300,seed=1):
    rng=np.random.default_rng(seed); w=res.model.var_weights if res.model.var_weights is not None else np.ones(res.model.exog.shape[0])
    X1=_X(res,var,1); X0=_X(res,var,0)
    draws=rng.multivariate_normal(res.params.values,res.cov_params().values,size=reps)
    return float(np.std([np.average(norm.cdf(X1@bb)-norm.cdf(X0@bb),weights=w) for bb in draws]))
def row(res,data,var='cc_full'):
    m=ame(res,data,var); s=ame_se(res,data,var)
    return {'coef':float(res.params[var]),'se':float(res.bse[var]),'p':float(res.pvalues[var]),'ame':m,'ame_se':s,'N':int(res.nobs)}
# VIF
Xv=a[CTRL].astype(float); Xv=sm.add_constant(Xv); R['vif_max']=max(variance_inflation_factor(Xv.values,i) for i in range(1,Xv.shape[1]))
# Main
T={}
T['adopt']=row(probit(f'ai_adopt ~ cc_full + {X} + C(cs)',a),a)
T['depth']=row(probit(f'ai_depth ~ cc_full + {X} + C(cs)',a),a)
ad=a[a.ai_adopt==1]; T['depth_adopters']=row(probit(f'ai_depth ~ cc_full + {X} + C(country)',ad),ad)
for v in ['ai_paid','ai_custom']:
    s=a.dropna(subset=[v]); T[v]=row(probit(f'{v} ~ cc_full + {X} + C(country)',s),s)
for v in ['ai_depth_uses','ai_depth_funcs']:
    T[v]=row(probit(f'{v} ~ cc_full + {X} + C(cs)',a),a)
R['main']=T
# IV
from linearmodels.iv import IV2SLS
ivr={}
for fe in ['country','cs']:
    s=a.dropna(subset=['iv_cc_full_loo','iv_bankfin_loo']).copy(); s['w']=s.wt_ai/s.wt_ai.mean()
    fs=smf.wls(f'cc_full ~ iv_cc_full_loo + {X} + C({fe})',s,weights=s.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s.cl)[0]})
    fs2=smf.wls(f'cc_full ~ iv_cc_full_loo + iv_bankfin_loo + {X} + C({fe})',s,weights=s.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s.cl)[0]})
    Fj=float(fs2.f_test('iv_cc_full_loo = 0, iv_bankfin_loo = 0').fvalue)
    # reduced form / AR test (just-identified)
    rf=smf.wls(f'ai_adopt ~ iv_cc_full_loo + {X} + C({fe})',s,weights=s.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s.cl)[0]})
    ivr[fe]={'F1':float(fs.tvalues['iv_cc_full_loo']**2),'pi':float(fs.params['iv_cc_full_loo']),'F2':Fj,'AR_p':float(rf.pvalues['iv_cc_full_loo']),'N':int(fs.nobs)}
# no-FE first stage (unconditional, controls only)
s=a.dropna(subset=['iv_cc_full_loo']).copy(); s['w']=s.wt_ai/s.wt_ai.mean()
fs0=smf.wls(f'cc_full ~ iv_cc_full_loo + {X}',s,weights=s.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s.cl)[0]})
ivr['none']={'F1':float(fs0.tvalues['iv_cc_full_loo']**2),'pi':float(fs0.params['iv_cc_full_loo'])}
# CF probit with country FE
s['vhat']=smf.wls(f'cc_full ~ iv_cc_full_loo + {X} + C(country)',s,weights=s.w).fit().resid
cf=probit(f'ai_adopt ~ cc_full + vhat + {X} + C(country)',s)
ivr['cf']={'coef':float(cf.params['cc_full']),'rho':float(cf.params['vhat']),'rho_p':float(cf.pvalues['vhat'])}
# Lewbel: generated instruments (Z-mean)*e from first stage of cc on X
s2=a.copy(); s2['w']=s2.wt_ai/s2.wt_ai.mean()
e=smf.wls(f'cc_full ~ {X} + C(country)',s2,weights=s2.w).fit().resid
lz=[]
for c in ['ln_emp','ln_age','mgr_exp','website','exporter']:
    s2['lw_'+c]=(s2[c]-s2[c].mean())*e; lz.append('lw_'+c)
ex=' + '.join(CTRL)+' + C(country)'
lew=IV2SLS.from_formula(f'ai_adopt ~ 1 + {ex} + [cc_full ~ {" + ".join(lz)}]',s2,weights=s2.w).fit(cov_type='clustered',clusters=pd.factorize(s2.cl)[0])
ivr['lewbel']={'b':float(lew.params['cc_full']),'se':float(lew.std_errors['cc_full']),'p':float(lew.pvalues['cc_full']),'F':float(lew.first_stage.diagnostics['f.stat'].iloc[0]),'J_p':float(lew.wooldridge_overid.pval) if hasattr(lew,'wooldridge_overid') else None}
try: ivr['lewbel']['sargan_p']=float(lew.sargan.pval)
except Exception as ex_: ivr['lewbel']['sargan_p']=None
lpm=smf.wls(f'ai_adopt ~ cc_full + {ex}',s2,weights=s2.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s2.cl)[0]})
ivr['ols_lpm']={'b':float(lpm.params['cc_full']),'se':float(lpm.bse['cc_full'])}
R['iv']=ivr
# Threshold
def hansen(data,yv,qv,reps=999,seed=7):
    y=data[yv].values.astype(float)
    FE=pd.get_dummies(data['cs'],drop_first=True,dtype=float).values
    W=np.column_stack([np.ones(len(data)),data[CTRL].values.astype(float),FE])
    q=data[qv].values.astype(float); cc=data.cc_full.values.astype(float)
    grid=np.unique(np.quantile(q,np.linspace(.15,.85,50)))
    def fit(yy,g):
        M=np.column_stack([W,cc*(q<=g),cc*(q>g)]); b,*_=np.linalg.lstsq(M,yy,rcond=None); e=yy-M@b; return e@e,b,M,e
    S=np.array([fit(y,g)[0] for g in grid]); k=int(np.argmin(S)); g=grid[k]; S1=S[k]
    M0=np.column_stack([W,cc]); b0,*_=np.linalg.lstsq(M0,y,rcond=None); e0=y-M0@b0; S0=e0@e0
    F=len(y)*(S0-S1)/S1
    rng=np.random.default_rng(seed); Fb=[]
    for _ in range(reps):
        yb=M0@b0+e0*rng.standard_normal(len(y)); bb,*_=np.linalg.lstsq(M0,yb,rcond=None); eb=yb-M0@bb
        Sb=min(fit(yb,gg)[0] for gg in grid); Fb.append(len(y)*(eb@eb-Sb)/Sb)
    LR=len(y)*(S-S1)/S1; ci=grid[LR<=7.35]
    _,b,M,e=fit(y,g)
    # HC1 SEs for regime coefs
    XtXi=np.linalg.pinv(M.T@M); V=XtXi@(M.T*(e**2))@M@XtXi*len(y)/(len(y)-M.shape[1]); se=np.sqrt(np.diag(V))
    nb=int(((q<=g)).sum())
    return {'tau':float(g),'F':float(F),'p':float(np.mean(np.array(Fb)>=F)),'ci':[float(ci.min()),float(ci.max())],'th1':float(b[-2]),'se1':float(se[-2]),'th2':float(b[-1]),'se2':float(se[-1]),'n_below':nb,'n_above':int(len(y)-nb),'N':len(y)}
t=a.dropna(subset=['comp_share_frac']).copy()
R['thr_depth']=hansen(t,'ai_depth','comp_share_frac')
R['thr_lpm']=hansen(t,'ai_adopt','comp_share_frac',reps=499)
# Gender
G={}
for gv in ['female_mgr','female_majority']:
    s=a.dropna(subset=[gv]).copy()
    m=probit(f'ai_adopt ~ cc_full*{gv} + {X} + C(country)',s)
    out={'int_coef':float(m.params[f'cc_full:{gv}']),'int_p':float(m.pvalues[f'cc_full:{gv}'])}
    gcol=m.model.exog[:,list(m.model.exog_names).index(gv)]
    for gg in [0,1]:
        out[f'ame_{gg}']=ame(m,mask=(gcol==gg)); out[f'n_{gg}']=int((gcol==gg).sum())
    # cluster bootstrap for difference
    rng=np.random.default_rng(3); cls=s.cl.unique(); diffs=[]; grp={c:g for c,g in s.groupby('cl')}
    for _ in range(299):
        pick=rng.choice(cls,len(cls),replace=True); bs=pd.concat([grp[c] for c in pick])
        try:
            mb=smf.glm(f'ai_adopt ~ cc_full*{gv} + {X} + C(country)',bs,family=Binomial(link=Probit()),var_weights=bs.wt_ai/bs.wt_ai.mean()).fit()
            gc=mb.model.exog[:,list(mb.model.exog_names).index(gv)]
            diffs.append(ame(mb,mask=(gc==1))-ame(mb,mask=(gc==0)))
        except Exception: pass
    out['diff']=out['ame_1']-out['ame_0']; out['diff_ci']=[float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]
    G[gv]=out
R['gender']=G
# Mechanism
na=a[(a.ai_adopt==0)].dropna(subset=['ai_cost_barrier'])
R['mech_cost']=row(probit(f'ai_cost_barrier ~ cc_full + {X} + C(country)',na),na)
pl=a.dropna(subset=['policy_ai_loans'])
R['mech_loans']=row(probit(f'policy_ai_loans ~ cc_full + {X} + C(country)',pl),pl)
# Robustness (adoption AME)
RB={}
RB['cc_any']=row(probit(f'ai_adopt ~ cc_any + {X} + C(cs)',a.dropna(subset=['cc_any'])),a.dropna(subset=['cc_any']),'cc_any')
for v in ['any_bank_finance','fin_major_obstacle']:
    s=a.dropna(subset=[v]); RB[v]=row(probit(f'ai_adopt ~ {v} + {X} + C(cs)',s),s,v)
s=a[a.no_need==0]; RB['demand']=row(probit(f'ai_adopt ~ cc_full + {X} + C(country)',s),s)
s=a[a.pre_adoption_sample==1]; RB['post_baseline']=row(probit(f'ai_adopt ~ cc_full + {X} + C(cs)',s),s)
s=a[a.country!='India']; RB['excl_india']=row(probit(f'ai_adopt ~ cc_full + {X} + C(cs)',s),s)
for c in a.country.unique():
    s=a[a.country==c]
    try: RB['c_'+c]=row(probit(f'ai_adopt ~ cc_full + {X} + C(sector_a4a)',s),s)
    except Exception as e_: RB['c_'+c]={'err':str(e_)}
lp=smf.wls(f'ai_adopt ~ cc_full + {X} + C(cs)',a,weights=a.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(a.cl)[0]})
RB['lpm']={'b':float(lp.params['cc_full']),'se':float(lp.bse['cc_full']),'p':float(lp.pvalues['cc_full'])}
uw=smf.glm(f'ai_adopt ~ cc_full + {X} + C(cs)',a,family=Binomial(link=Probit())).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(a.cl)[0]})
a1=a.copy(); a1['wt_ai']=1.0
RB['unweighted']={'coef':float(uw.params['cc_full']),'p':float(uw.pvalues['cc_full']),'ame':ame(uw)}
# Oster delta (LPM, Rmax=1.3R)
short=smf.wls('ai_adopt ~ cc_full',a,weights=a.w).fit(); long_=smf.wls(f'ai_adopt ~ cc_full + {X} + C(cs)',a,weights=a.w).fit()
b0,R0=short.params['cc_full'],short.rsquared; b1,R1=long_.params['cc_full'],long_.rsquared; Rm=min(1.3*R1,1)
delta=(b1*(R1-R0))/((b0-b1)*(Rm-R1)) if b0!=b1 else np.inf
bstar=b1-1*(b0-b1)*(Rm-R1)/(R1-R0)
RB['oster']={'b_short':float(b0),'b_long':float(b1),'R0':float(R0),'R1':float(R1),'delta':float(delta),'beta_star':float(bstar)}
t2=a.copy(); t2['web']=t2.website
# website threshold isn't continuous; use interaction instead
m=probit(f'ai_depth ~ cc_full*website + {X} + C(cs)',a)
wc=m.model.exog[:,list(m.model.exog_names).index('website')]
RB['website_int']={'coef':float(m.params['cc_full:website']),'p':float(m.pvalues['cc_full:website']),'ame_web0':ame(m,mask=wc==0),'ame_web1':ame(m,mask=wc==1)}
R['robust']=RB
json.dump(R,open('results.json','w'),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))

# ---- Country-normalised weights (Table 8) ----
a2=a.copy(); a2['wt_ai']=a2.wt_ai/a2.groupby('country').wt_ai.transform('sum')
CN={'adopt':row(probit(f'ai_adopt ~ cc_full + {X} + C(cs)',a2),a2),'depth':row(probit(f'ai_depth ~ cc_full + {X} + C(cs)',a2),a2)}
s3=a2.dropna(subset=['ai_custom']); CN['custom']=row(probit(f'ai_custom ~ cc_full + {X} + C(country)',s3),s3)
print(json.dumps(CN,indent=1))
