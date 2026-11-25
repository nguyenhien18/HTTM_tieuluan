
demoReady.then(()=>{
  main.innerHTML=`${head()}<div class="split"><section class="panel"><div class="panel-head"><div><h3>Đồ thị thực thể và quan hệ</h3><p>Chọn một nút để xem bằng chứng liên quan.</p></div><span class="badge-live">14 thực thể</span></div><div class="graph-wrap">${graphHtml(state.config.graph)}</div><p class="graph-caption">Di chuột lên cạnh để xem quan hệ và mã nguồn. Bấm nút trong đồ thị để đọc tài liệu.</p></section>${panel('Tài liệu của nút đã chọn',`<div class="panel-body" id="related-documents"><p class="hint">Chọn một thực thể trong đồ thị bên trái để bắt đầu.</p></div>`)}</div>`;
  bindGraph();
});

