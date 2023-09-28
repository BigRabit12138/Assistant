from agents.messages import SceneMessage

msg1 = SceneMessage(**{'role': '李三',
                       'action': '张三拿起一米长的大刀。',
                       'conversation': '我的大刀早已饥渴难耐了.'}
                    )
msg2 = SceneMessage(role='李三',
                    action='张三拿起一米长的大刀。',
                    conversation='我的大刀早已饥渴难耐了.')
msg3 = SceneMessage(role='director', action='无尽虚空。')
print(msg1)
print(msg2)
print(msg3)
pass

