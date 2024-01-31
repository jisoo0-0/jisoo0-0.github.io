---
layout: post
categories: 꿀팁
tags: 꿀팁
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[꿀팁] 오류 빨리 확인하기"
---

1 epoch 도는데 너무 오래걸려서 그 후에 발생하는 오류를 확인하는데까지 시간이 너무 오래 걸릴 때 사용하면 좋은 방법

- data loader를 아래처럼 수정해주면 output시켜주는 datasize의 크기를 조절해줘서 빨리 확인하고 빨리 해결할 수 있음
    
    ```python
    def __len__(self):
            return 10 #10말고 1 등등 작은 숫자 아무거나 해도 무관함. 
    #당연하겠지만 오류 수정 후에는 원래 길이로 수정 필요
    ```

간단하지만 유용한.. ㅋㅋ

<br/>
<br/>
<div id="disqus_thread"></div>
<script>
    (function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://https-jisoo0-0-github-io.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/