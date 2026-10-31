"""A single live plot output under the notebook's running training cell."""
from collections import OrderedDict
from io import BytesIO
import time
from .paths import ROOT

class InlineTrainingPlot:
    def __init__(self,total,workers,fresh=False):
        from IPython.display import display,Image
        self.total=total;self.workers=workers;self.curves=OrderedDict();self.last=0
        self.completed=0 if fresh else len(list((ROOT/'results/replication/checkpoints').glob('*/*/trial_*.json')))
        self.display=display;self.Image=Image;self.handle=None
        self.draw(force=True)

    def update(self,event):
        key=(event['strategy'],event['fold'],event['trial'])
        # One current candidate per model/window worker. Retain its real epochs.
        if event['kind']=='epoch':
            for old in list(self.curves):
                if old[:2]==key[:2] and old!=key:del self.curves[old]
            if key not in self.curves:
                while len(self.curves)>=self.workers:self.curves.popitem(last=False)
                self.curves[key]=[]
            self.curves[key].append(event)
            self.curves.move_to_end(key)
        elif event['kind']=='candidate_completed':
            self.completed=len(list((ROOT/'results/replication/checkpoints').glob('*/*/trial_*.json')))
        self.draw()

    def draw(self,force=False,finished=False):
        if not force and time.monotonic()-self.last<2:return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(11,7),dpi=100,layout='constrained');FigureCanvasAgg(fig)
        axes=fig.subplots(2,2).ravel();ax=axes[0]
        ax.barh([0],[self.total],color='#e6edf3',height=.35)
        ax.barh([0],[self.completed],color='#168a70',height=.35)
        ax.set_xlim(0,self.total);ax.set_ylim(-.6,.6);ax.set_yticks([])
        ax.set_title(f'Fresh model training: {self.completed:,} / {self.total:,} candidates',loc='left',fontsize=12)
        ax.set_xlabel('Completed independent candidate fits')
        ax.text(.01,.08,'Finished' if finished else 'Real epoch updates • no test-set metrics',transform=ax.transAxes,color='#52697a',fontsize=10)
        for ax,(key,points) in zip(axes[1:],self.curves.items()):
            strategy,fold,trial=key
            ax.plot([p['epoch'] for p in points],[p['training_loss'] for p in points],label='Training',color='#2563a9')
            ax.plot([p['epoch'] for p in points],[p['validation_loss'] for p in points],label='Validation',color='#cf7b16')
            ax.set(title=f'{strategy} | test {fold} | candidate {trial+1}',xlabel='Epoch',ylabel='Negative Sharpe loss')
            ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.2)
        for ax in axes[1+len(self.curves):]:
            ax.text(.5,.5,'Waiting for epoch records',ha='center',va='center',transform=ax.transAxes,color='#627689');ax.set_axis_off()
        fig.suptitle('Paper 67 — live training / validation curves',fontsize=15)
        output=BytesIO();fig.savefig(output,format='png');data=output.getvalue()
        if self.handle is None:self.handle=self.display(self.Image(data=data),display_id=True)
        else:self.handle.update(self.Image(data=data))
        self.last=time.monotonic()
        if finished:
            dest=ROOT/'results/replication/figures/training_monitor.png';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
