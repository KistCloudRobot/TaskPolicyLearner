from Lib.ILPRLEngine import *
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.CONJ import CONJ


class ILP_MODEL(object):
    def __init__(self, num_box,is_train=True):
        self.num_box = num_box
        self.args = self.load_ilp_config()
        self.define_preds()
        self.Xo=None
        self.X0=None

    def load_ilp_config(self):
        
        param = dotdict({})
        param.BS = 1
        param.T = 1
        param.W_DISP_TH = .1
        param.GPU = 1
         
        return param 
        

    def define_preds(self):  

        '''

        상수 선언 및 서술자 선언 필요

        '''

        Robot = ['amr-lift01']  #, 'amr-lift02','amr-lift03', 'amr-lift04']
        Task = ['transport']
        Grade =['1', '2', '3', '4', '5']

        self.Constants = dict( {'R':Robot,'T':Task})
        self.predColl = PredCollection (self.Constants)

        #self.predColl.add_pred(dname='is_robot', arguments=['R'])
        #self.predColl.add_pred(dname='is_task', arguments=['T'])

        ############## cloud
        self.predColl.add_pred(dname='safety_B1', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_C1', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_L1', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])

        self.predColl.add_pred(dname='safety_B2', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_C2', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_L2', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])

        self.predColl.add_pred(dname='safety_B3', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_C3', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_L3', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])

        self.predColl.add_pred(dname='safety_B4', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_C4', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_L4', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])

        self.predColl.add_pred(dname='safety_B5', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_C5', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        self.predColl.add_pred(dname='safety_L5', arguments=['R', 'T'],exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])


        #self.predColl.add_pred(dname='safety_grade_comparison', arguments=['G', 'G', 'G', 'G'])
        pt=[('and','safety_B1(A,B)')]
        pt = []
        init1 = list()
        init2 = list()
        init3 = list()
        init4 = list()
        init5 = list()

        for i in range(1,6):
            for j in range(1,6):
                for k in range(1,6):
                    s_grade = round((i+j+k)/3)
                    if s_grade == 1:
                        init1.append("safety_B"+str(i)+"(A,B), safety_C"+str(j)+"(A,B), safety_L"+str(k)+"(A,B)")
                    elif s_grade == 2:
                        init2.append("safety_B"+str(i)+"(A,B), safety_C"+str(j)+"(A,B), safety_L"+str(k)+"(A,B)")
                    elif s_grade == 3:
                        init3.append("safety_B"+str(i)+"(A,B), safety_C"+str(j)+"(A,B), safety_L"+str(k)+"(A,B)")
                    elif s_grade == 4:
                        init4.append("safety_B"+str(i)+"(A,B), safety_C"+str(j)+"(A,B), safety_L"+str(k)+"(A,B)")
                    elif s_grade == 5:
                        init5.append("safety_B"+str(i)+"(A,B), safety_C"+str(j)+"(A,B), safety_L"+str(k)+"(A,B)")


        self.predColl.add_pred(dname='safety1', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety1', terms=4, init=[1, -1, -1, .1], sig=2,
             init_terms=init1, predColl=self.predColl,
            fast=True), use_neg=False, Fam='eq', exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        
 
        self.predColl.add_pred(dname='safety2', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety2', terms=31, init=[1, -1, -1, .1], sig=2,
             init_terms=init2, predColl=self.predColl,
            fast=True), use_neg=False, Fam='eq', exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        


        self.predColl.add_pred(dname='safety3', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety3', terms=55, init=[1, -1, -1, .1], sig=2,
             init_terms=init3, predColl=self.predColl,
            fast=True), use_neg=False, Fam='eq', exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        

        self.predColl.add_pred(dname='safety4', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety4', terms=31, init=[1, -1, -1, .1], sig=2,
             init_terms=init4, predColl=self.predColl,
            fast=True), use_neg=False, Fam='eq', exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        


        self.predColl.add_pred(dname='safety5', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety5', terms=4, init=[1, -1, -1, .1], sig=2,
             init_terms=init5, predColl=self.predColl,
            fast=False), use_neg=False, Fam='eq', exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
        




        '''
        
                self.predColl.add_pred(dname='safety1', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety1', terms=4, init=[1, -1, -1, .1], sig=2,
             init_terms=['safety_B1(A,B), safety_C1(A,B), safety_L1(A,B)', 'safety_B2(A,B), safety_C1(A,B), safety_L1(A,B)', 'safety_B1(A,B), safety_C2(A,B), safety_L1(A,B)', 'safety_B1(A,B), safety_C1(A,B), safety_L2(A,B)'], predColl=self.predColl,
            fast=True), use_neg=False, Fam='eq', exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
            
            
            
        self.predColl.add_pred(dname='safety5', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety5', terms=4, init=[1, -1, -1, .1], sig=2,
             init_terms=['safety_B5(A,B), safety_C5(A,B), safety_L5(A,B)', 'safety_B4(A,B), safety_C5(A,B), safety_L5(A,B)', 'safety_B5(A,B), safety_C4(A,B), safety_L5(A,B)', 'safety_B5(A,B), safety_C5(A,B), safety_L4(A,B)'], predColl=self.predColl,
            fast=True), use_neg=False, Fam='eq', exc_preds=['safety1', 'safety2', 'safety3','safety4', 'safety5'])
            

        self.predColl.add_pred(dname='safety1', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety1', terms=4, init=[1, -1, -1, .1], sig=2,
             predColl=self.predColl,post_terms=pt,
            fast=False), use_neg=True, Fam='eq',exc_preds=[], exc_conds=[('*','rep1')])

        self.predColl.add_pred(dname='safety2', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety2', terms=4, init=[1, -1, -1, .1], sig=2,
             predColl=self.predColl,
            fast=False), use_neg=True, Fam='eq',exc_preds=[], exc_conds=[('*','rep1')])

        self.predColl.add_pred(dname='safety3', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety3', terms=4, init=[1, -1, -1, .1], sig=2,
             predColl=self.predColl,
            fast=False), use_neg=True, Fam='eq',exc_preds=[], exc_conds=[('*','rep1')])

        self.predColl.add_pred(dname='safety4', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety4', terms=4, init=[1, -1, -1, .1], sig=2,
             predColl=self.predColl,
            fast=False), use_neg=True, Fam='eq',exc_preds=[], exc_conds=[('*','rep1')])

        self.predColl.add_pred(dname='safety5', arguments=['R', 'T'], variables=[], pFunc=
        DNF('safety5', terms=4, init=[1, -1, -1, .1], sig=2,
             predColl=self.predColl,
            fast=False), use_neg=True, Fam='eq',exc_preds=[], exc_conds=[('*','rep1')])

       '''

        self.predColl.initialize_predicates()
        self.bg = Background(self.predColl)
        #####define background###
        #self.bg.add_backgroud('is_robot', ('amr-lift01',))
        #self.bg.add_backgroud('is_robot', ('amr-lift02',))
        #self.bg.add_backgroud('is_robot', ('amr-lift03',))
        #self.bg.add_backgroud('is_robot', ('amr-lift04',))
        #self.bg.add_backgroud('is_task', ('transport',))
        print('displaying config setting...')
        self.mdl = ILPRLEngine( args=self.args ,predColl=self.predColl, bgs=None )


    def run(self, x, x1, x2):
        bs = tf.shape(x)[0]
        self.X0=OrderedDict()
        #print("Xs", x, x1, x2, x3, x4)

        for p in self.predColl.outpreds:
            tmp = tf.expand_dims(tf.constant(self.bg.get_X0(p.oname), tf.float32), 0)
            self.X0[p.oname] = tf.tile(tmp, [bs, 1])

        #print(self.X0['safety_C'])
        #print(self.X0['safety_B'])
        #print(self.X0['efficiency_T'])
        #print(self.X0['efficiency_B'])
        
        #print("####",tf.reshape(x2[4],[1,5]))


        self.X0['safety_C1'], self.X0['safety_C2'], self.X0['safety_C3'], self.X0['safety_C4'], self.X0['safety_C5'] = tf.split(x,num_or_size_splits=5, axis=1)
        
        self.X0['safety_B1'], self.X0['safety_B2'], self.X0['safety_B3'], self.X0['safety_B4'], self.X0['safety_B5'] = tf.split(x1,num_or_size_splits=5, axis=1)
        
        self.X0['safety_L1'], self.X0['safety_L2'], self.X0['safety_L3'], self.X0['safety_L4'], self.X0['safety_L5'] = tf.split(x2,num_or_size_splits=5, axis=1)


        self.Xo, L3 = self.mdl.getTSteps(self.X0)


        #action = [self.Xo[i] for i in ['moveToLoad', 'load', 'unload', 'carry', 'returnRack']]

        action = [self.Xo[i] for i in ['safety1', 'safety2', 'safety3', 'safety4', 'safety5']]


        return tf.concat(action,-1), self.Xo

