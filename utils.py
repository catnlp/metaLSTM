# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/24 23:57
'''
def clip_grad(v, min, maz):
    v_tmp = v.expand_as(v)
    v_tmp.register_hook(lambda g: g.clamp(min, max))
    return v_tmp