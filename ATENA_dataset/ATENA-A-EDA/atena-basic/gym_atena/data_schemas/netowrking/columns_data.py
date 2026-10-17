KEYS = ['packet_number', 'eth_dst', 'eth_src', 'highest_layer', 'info_line',
       'ip_dst', 'ip_src', 'length',
        'sniff_timestamp', 'tcp_dstport', 'tcp_srcport',
       'tcp_stream']
KEYS_ANALYST_STR = ['Packet No.', 'Dest MAC', 'Source MAC', 'Protocol', 'Info',
       'Dest IP', 'Source IP', 'Length',
        'TimeStamp', 'Dest Port', 'SRC Port',
       'TCP Session']
FILTER_COLS = KEYS
GROUP_COLS = KEYS
NUMERIC_KEYS = {'packet_number', 'length', 'tcp_dstport', 'tcp_srcport', 'tcp_stream'}


NUMERIC_KEYS = {'length'}
AGG_KEYS = ['packet_number']
AGG_KEYS_ANALYST_STR = ['Packet No.']
FILTER_LIST = ['82.108.87.7',
               '82.108',
               '75',
               '82.108.163.88',
               'login',
               'javascript',
               '00:0c:29:54:bf:79',
               'PSH',
               'reply',
               'reply id=0x0200',
               'BROWSER',
               'applicatio',
               '00:26:b9:2b:0b:59',
               'Echo (ping) reply',
               'SYN, ACK',
               'SYN ACK',
               'LLENGE, Error: STATUS_MORE_PROCE',
               'Echo (ping) reply id',
               'NBNS',
               '42.253',
               '28',
               '82.',
               '135',
               'ping',
               'SMB',
               '82.108.84.10',
               '10.0.4.15',
               '[SYN, ACK]',
               '443',
               '192',
               '10.0.3.15',
               'Auth',
               'BOOTP',
               'auth',
               '8884',
               'SOCKS',
               '64.236.114.1',
               '82.108.87.6',
               '98.114.205.102',
               '82.108.24.168',
               '139',
               'SYN',
               '.56.52',
               '82.108.203.177',
               'HTTP',
               'RST, ACK',
               'exe',
               'Nzmx',
               '80',
               '82.108.10.107',
               'ICMP',
               '192.169.1.122',
               '17',
               '1370168419',
               'request',
               'fg',
               '192.168.1.122',
               '200',
               'Error:',
               'Not Found',
               'DATA-TEXT-LINES',
               '10.42',
               '1957',
               'DHCP',
               '82.108.0.41',
               'GET',
               '255.255.255.255',
               '94.114.205.102',
               '82.108.19.97',
               'load.php?e=1',
               '192.168.1.1',
               'icmp',
               'favicon',
               'SYN,',
               '10.0.2.15',
               '82.108.163.80',
               '121',
               'reply id',
               '3',
               '192.168.56.50',
               'Echo (ping) request',
               '43',
               'ARP',
               'tcp',
               'MEDIA',
               '1331',
               '2',
               '445',
               'honey',
               '82.108.171.155',
               '1',
               'http',
               '82.108.69',
               '1370168368',
               'ACK',
               'Who has',
               'TCP',
               '0',
               'POST',
               'destination',
               '[SYN, ACK]',
               '10.42.42.253',
               '2152',
               '10.0.2.2',
               '10.0.',
               '3feb5a6b2f',
               '82.108.56.65',
               '10.42.42.50',
               '5',
               '62',
               'Ja',
               'MZ',
               '82.108.10.174',
               '192.168.56.52',
               'RST',
               'Echo',
               'SYN,ACK',
               '[RST, ACK]',
               '[SYN , ACK]'
               ]


FILTER_BY_FIELD_DICT = {'packet_number': set(), 'eth_dst': set(), 'eth_src': set(),
                        'highest_layer': set(), 'info_line': set(),
                        'ip_dst': set(), 'ip_src': set(), 'length': set(), 'number': set(),
                        'sniff_timestamp': set(), 'tcp_dstport': set(), 'tcp_srcport': set(),
                        'tcp_stream': set()}
DONT_FILTER_FIELDS = {'sniff_timestamp', 'packet_number', 'length', 'number'}

FILTER_BY_FIELD_DICT['eth_dst'] = {'00:0c:29:54:bf:79', '00:26:b9:2b:0b:59'}
FILTER_BY_FIELD_DICT['eth_src'] = {'00:0c:29:54:bf:79', '00:26:b9:2b:0b:59', '52:54:00:12:35:00'}
FILTER_BY_FIELD_DICT['highest_layer'] = {'SMB',

                                         'HTTP','ICMP',
                                         'icmp',

                                         'tcp', 'TCP',
                                         'http', 'HTTP',

                                         'SOCKS',


                                         'DSSETUP',
                                         'DCERPC',

                                         }
FILTER_BY_FIELD_DICT['info_line'] = {'[SYN, ACK]',
                                     'ACK', 'SYN,ACK', '[RST, ACK]', '[SYN , ACK], RST', '[RST]'
                                     'SYN,', 'RST, ACK', 'SYN', '[SYN, ACK]',
                                     'SYN, ACK', 'SYN ACK', 'Echo (ping) reply',
                                     'Echo (ping) reply id', 'reply',
                                     'reply id=0x0200',

                                     'Echo (ping) request', 'reply id',
                                     'Echo', 'POST', 'GET', 'destination',
                                     '200', 'login',

                                     'PSH',


                                     'ping', 'load.php?e=1', 'exe',

                                     '[FIN, PSH, URG]',
                                     'URG',
                                     '[SYN]',
                                     'Destination'
                                     }
FILTER_BY_FIELD_DICT['ip_dst'] = {
    '82.108.87.7', '82.108', '82.108.163.88', '42.253', '82.108.56.65', '192.168.1.122',
    '10.42.42.50', '82.108.10.174', '192.168.56.52', '82.108.171.155',
    '82.108.10.107', '10.42.42.253', '82.108.69', '192.168.56.50', '10.0.2.15',
    '82.108.163.80', '192.168.1.1', '255.255.255.255', '94.114.205.102',
    '82.108.19.97', '82.108.0.41', '192.169.1.122', '82.108.203.177',

    '82.108.87.6', '98.114.205.102', '82.108.24.168',
    '.56.52', '10.0.3.15', '82.108.84.10', '10.0.4.15', '82.', '10.0.2.2',
    '10.0.', '192', '10.42',
}
FILTER_BY_FIELD_DICT['ip_src'] = {
    '82.108.87.7', '82.108', '82.108.163.88', '42.253', '82.108.56.65', '192.168.1.122',
    '10.42.42.50', '82.108.10.174', '192.168.56.52', '82.108.171.155',
    '82.108.10.107', '10.42.42.253', '82.108.69', '192.168.56.50', '10.0.2.15',
    '82.108.163.80', '192.168.1.1', '255.255.255.255', '94.114.205.102',
    '82.108.19.97', '82.108.0.41', '192.169.1.122', '82.108.203.177',

    '82.108.87.6', '98.114.205.102', '82.108.24.168',
    '.56.52', '10.0.3.15', '82.108.84.10', '10.0.4.15', '82.', '10.0.2.2',
    '10.0.', '192', '10.42',
}
FILTER_BY_FIELD_DICT['sniff_timestamp'] = {'1370168419', '1370168368'}
FILTER_BY_FIELD_DICT['tcp_dstport'] = {135, 443, 139, 80,
                                       8884, 445,

                                       1331,
                                       }
FILTER_BY_FIELD_DICT['tcp_srcport'] = {135, 443, 139, 80,
                                       8884, 445, 2152, 1331,
                                       }
FILTER_BY_FIELD_DICT['tcp_stream'] = {0,
                                      1,

                                      3,
                                      5,
                                      17,
                                      43,
                                      121,
                                      }
