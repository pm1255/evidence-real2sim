"""Audit and run available numeric branches. Optional images/assets run explicitly.

This orchestrator does not call physical robots, install packages, download data
or silently substitute a robot/asset. Results go to a fresh output directory.
"""
import argparse
import html
from pathlib import Path
import subprocess
import sys
from common import fresh,read,dump,digest


ROOT=Path(__file__).resolve().parent


def run(manifest,out,contact_config=None,torque_config=None,evaluate_test=False):
    out=fresh(out);manifest=Path(manifest).resolve();steps=[]
    commands=[('audit',['02_audit_inputs.py','--manifest',str(manifest),'--out',str(out/'audit.json')]),
              ('controller',['05_fit_controller.py','--manifest',str(manifest),'--out',str(out/'controller')])]
    if contact_config:commands.append(('contact_fit',['06_fit_contact.py','fit','--manifest',str(manifest),'--config',str(Path(contact_config).resolve()),'--out',str(out/'contact_fit')]))
    if torque_config:commands.append(('torque',['07_torque_residual.py','--manifest',str(manifest),'--config',str(Path(torque_config).resolve()),'--out',str(out/'torque')]))
    for name,args in commands:
        p=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],capture_output=True,text=True)
        (out/(name+'.log')).write_text(p.stdout+p.stderr)
        steps.append({'stage':name,'exit_code':p.returncode,'log':name+'.log'})
        if name=='audit' and p.returncode:break
    if evaluate_test and contact_config and (out/'contact_fit/contact_model.json').exists():
        fitted=read(out/'contact_fit/contact_model.json')
        if fitted.get('status')!='skipped':
            cmd=[sys.executable,str(ROOT/'06_fit_contact.py'),'evaluate','--manifest',str(manifest),
                 '--config',str(Path(contact_config).resolve()),'--model',str(out/'contact_fit/contact_model.json'),
                 '--out',str(out/'contact_test'),'--split','test']
            p=subprocess.run(cmd,capture_output=True,text=True);(out/'contact_test.log').write_text(p.stdout+p.stderr)
            steps.append({'stage':'contact_test','exit_code':p.returncode,'log':'contact_test.log'})
    result={'manifest_sha256':digest(manifest),'steps':steps,
            'script_sha256':{p.name:digest(p) for p in ROOT.glob('*.py')},
            'data_origin':read(manifest).get('data_origin'),'test_evaluation_requested':evaluate_test,
            'real_sim_policy_equivalence_validated':False,
            'note':'Successful process exit is not proof of contact or evaluation fidelity.'}
    dump(out/'run.json',result)
    cards=''.join(f'<tr><td>{html.escape(s["stage"])}</td><td>{"完成运行" if s["exit_code"]==0 else "失败，查看日志"}</td><td><a href="{s["log"]}">日志</a></td></tr>' for s in steps)
    (out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>按数据条件运行的仿真校准</title><style>body{font-family:system-ui;max-width:960px;margin:48px auto;padding:0 24px;line-height:1.8;color:#192c38}td,th{padding:10px 24px;border-bottom:1px solid #ddd}aside{background:#fff3d5;padding:16px}a{color:#166e80}</style><h1>按数据条件运行的仿真校准</h1><p>数据来源：'+html.escape(str(result['data_origin']))+'</p><aside>本页记录软件执行情况。接触参数是否准确、仿真是否能替代真机，需要独立实验支持；不会因脚本运行成功自动通过。</aside><table><tr><th>阶段</th><th>状态</th><th>详情</th></tr>'+cards+'</table><p><a href="run.json">完整运行记录</a> · <a href="audit.json">数据缺失与能力审计</a> · <a href="controller/controller.json">控制响应结果</a></p></html>')
    print(f'Wrote {out}/report.html')
    if any(s['exit_code'] for s in steps):raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--contact-config');p.add_argument('--torque-config');p.add_argument('--evaluate-test',action='store_true')
    a=p.parse_args();run(a.manifest,Path(a.out).resolve(),a.contact_config,a.torque_config,a.evaluate_test)


if __name__=='__main__':main()
