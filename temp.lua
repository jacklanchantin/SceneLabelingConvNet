                

                yMap =1;xMap = 1; 
                start_pixel = 7;
                step_pixel = 2
                X = 5
                Y = 5
                x = torch.Tensor(X,Y)
                s = x:storage()                                                                                                   
                for i=1,s:size() do
                  s[i] = i 
                end
                labels = {}
                k = 0
                for i=1,X do
                    labels[i] = {}
                    for j=1,Y do
                        k = k+1
                        labels[i][j] = k
                    end
                end

                for xMap=0,step_pixel-1 do
                    for yMap=0,step_pixel-1 do
                        paddedImg = torch.zeros(x:size(1)+((start_pixel-1)*2)-(yMap), x:size(2)+((start_pixel-1)*2)-(xMap))
                        xS = (start_pixel)-yMap
                        xF = x:size(1)+(start_pixel)-(yMap)-1
                        yS = (start_pixel)-xMap
                        yF = x:size(2)+(start_pixel)-(xMap)-1
                        paddedImg[{{xS,xF},{yS,yF}}] = x
                        print(paddedImg)
                    end
                end


                ans = {}
                k = 0
                for i=1,x:size(1)-yMap,step_pixel do
                    for j=1,x:size(2)-xMap,step_pixel do
                        k = k + 1
                        ans[k] = labels[i+xMap][j+yMap]
                    end
                end
                print(ans)
                print(k)