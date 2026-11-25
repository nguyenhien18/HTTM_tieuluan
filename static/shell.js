'use strict';
(() => {
  const get=id=>document.getElementById(id);
  get('help-button').onclick=()=>get('help-dialog').showModal();
  get('close-help').onclick=get('got-it').onclick=()=>get('help-dialog').close();
  get('help-dialog').onclick=e=>{if(e.target===get('help-dialog'))get('help-dialog').close()};
  get('menu-button').onclick=()=>document.querySelector('.sidebar').classList.toggle('open');
  get('export-button').onclick=()=>{
    const result=window.lastDemoResult;if(!result)return;
    const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json;charset=utf-8'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=`HTTM-${window.DEMO_PAGE?.id||'demo'}-${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
})();
