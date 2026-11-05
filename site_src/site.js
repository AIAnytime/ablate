(function(){
  var root=document.documentElement, $=function(s){return document.querySelector(s)}, $$=function(s){return document.querySelectorAll(s)};
  var th=$('#theme');
  if(th)th.addEventListener('click',function(){
    var t=root.getAttribute('data-theme')==='light'?'dark':'light';root.setAttribute('data-theme',t);
    try{localStorage.setItem('ablate-theme',t)}catch(e){}
  });
  function copy(text,btn){
    var done=function(){var o=btn.innerHTML;btn.textContent='✓';setTimeout(function(){btn.innerHTML=o},1200)};
    if(navigator.clipboard){navigator.clipboard.writeText(text).then(done,function(){})}
  }
  $$('[data-copy]').forEach(function(b){b.addEventListener('click',function(){copy($(b.dataset.copy).textContent,b)})});
  $$('[data-tabs]').forEach(function(box){
    var tabs=box.querySelectorAll('[data-t]'),panes=box.querySelectorAll('[data-p]');
    tabs.forEach(function(t){t.addEventListener('click',function(){
      tabs.forEach(function(x){x.setAttribute('aria-selected',x===t)});
      panes.forEach(function(p){p.hidden=p.dataset.p!==t.dataset.t});
    })});
    var c=box.querySelector('[data-copy-active]');
    if(c)c.addEventListener('click',function(e){copy(box.querySelector('[data-p]:not([hidden])').textContent,e.currentTarget)});
  });
  // typewriter demo (toolkit page)
  (function(){
    var run=$('#runmock');
    if(!run||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
    var el={},full={},tags=$$('[data-tag]'),frame=run.closest('.frame'),timer,started=false;
    $$('[data-type]').forEach(function(e){var k=e.dataset.type;full[k]=e.textContent;el[k]=e;
      e.innerHTML='<span class="on"></span><span class="ghost"></span>';e.querySelector('.ghost').textContent=full[k]});
    function set(k,n,caret){var e=el[k];e.firstChild.textContent=full[k].slice(0,n);e.firstChild.classList.toggle('caret',!!caret);e.lastChild.textContent=full[k].slice(n)}
    function type(k,step,ms,done){var n=0;(function tick(){n=Math.min(full[k].length,n+step);set(k,n,n<full[k].length);
      if(n<full[k].length)timer=setTimeout(tick,ms+Math.random()*ms);else done&&done()})()}
    function play(){
      ['p','a','b'].forEach(function(k){set(k,0,false)});tags.forEach(function(t){t.classList.add('off')});
      type('p',1,34,function(){
        timer=setTimeout(function(){run.classList.add('press');setTimeout(function(){run.classList.remove('press')},180);
          type('a',2,26,function(){tags[0].classList.remove('off')});
          type('b',3,24,function(){tags[1].classList.remove('off');timer=setTimeout(play,6000)});
        },450);
      });
    }
    function start(){if(started)return;started=true;set('p',0,true);timer=setTimeout(play,500)}
    if('IntersectionObserver' in window){new IntersectionObserver(function(en,o){if(en[0].isIntersecting){start();o.disconnect()}},{threshold:.35}).observe(frame)}else start();
  })();
  // screenshot gallery (toolkit page)
  var g=$$('[data-g]'),img=$('#gimg');
  g.forEach(function(b){b.addEventListener('click',function(){
    g.forEach(function(x){x.setAttribute('aria-selected',x===b)});
    img.src=img.src.replace(/[^\/]+\.webp$/,b.dataset.g+'.webp');img.alt=b.dataset.alt;img.removeAttribute('height');img.parentNode.scrollTop=0;
  })});
})();
