"""Loopback-only JSON HTTP/SSE facade. Operator and B credentials are separate."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import secrets
import time
from urllib.parse import parse_qs, urlsplit

from coordination.operator_common import OperatorError, canonical, require


class OperatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager, token, origins=()):
        require(address[0]=='127.0.0.1', 'invalid_bind', 'Operator API must bind IPv4 loopback')
        require(isinstance(token,str) and len(token)>=32, 'weak_token', 'Need a random operator credential')
        self.manager, self.token, self.origins = manager, token, frozenset(origins)
        super().__init__(address, OperatorHandler)
        self.manager.url = f'http://127.0.0.1:{self.server_port}/api/v1'


class OperatorHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(5.)

    def log_message(self, fmt, *args):
        pass  # URLs/credentials are never logged.

    def _json(self, code, payload):
        data = canonical(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        origin = self.headers.get('Origin')
        if origin in self.server.origins:
            self.send_header('Access-Control-Allow-Origin',origin)
            self.send_header('Vary','Origin')
        self.end_headers()
        self.wfile.write(data)

    def _security(self):
        port = self.server.server_port
        require(self.headers.get('Host') in {f'127.0.0.1:{port}',f'localhost:{port}'},
                'invalid_host', 'Loopback Host required', 403)
        origin = self.headers.get('Origin')
        require(origin is None or origin in self.server.origins, 'origin_denied', 'Origin not allowed', 403)

    def _token(self):
        value = self.headers.get('Authorization','')
        return value[7:] if value.startswith('Bearer ') else ''

    def _body(self):
        require(not self.headers.get('Transfer-Encoding'), 'invalid_body', 'Chunked requests not supported')
        require(self.headers.get_content_type()=='application/json', 'invalid_content_type', 'Use application/json', 415)
        lengths = self.headers.get_all('Content-Length',[])
        require(len(lengths)==1 and lengths[0].isdigit(), 'invalid_length', 'One Content-Length required')
        length = int(lengths[0])
        require(0 < length <= 65536, 'body_too_large', 'Maximum request body is 64 KiB', 413)
        try:
            def unique(pairs):
                result = {}
                for k,v in pairs:
                    if k in result:
                        raise ValueError('duplicate JSON field')
                    result[k]=v
                return result
            result = json.loads(self.rfile.read(length), object_pairs_hook=unique,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        except (ValueError, UnicodeDecodeError) as exc:
            raise OperatorError('invalid_json','Invalid JSON object') from exc
        require(isinstance(result,dict), 'invalid_body', 'JSON object required')
        return result

    def do_OPTIONS(self):
        try:
            self._security()
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin',self.headers.get('Origin',''))
            self.send_header('Access-Control-Allow-Methods','GET,POST,PUT,OPTIONS')
            self.send_header('Access-Control-Allow-Headers','Authorization,Content-Type')
            self.send_header('Content-Length','0')
            self.end_headers()
        except OperatorError as exc:
            self._json(exc.status,dict(error=exc.code,message=str(exc)))

    def do_GET(self): self._dispatch()
    def do_POST(self): self._dispatch()
    def do_PUT(self): self._dispatch()

    def _dispatch(self):
        self.close_connection = True
        try:
            self._security()
            target = urlsplit(self.path)
            path = target.path
            manager = self.server.manager
            if path.startswith('/api/v1/internal/'):
                return self._internal(path)
            require(secrets.compare_digest(self._token(),self.server.token), 'unauthorized','Operator credential required',401)
            if self.command=='GET' and path=='/api/v1/capabilities':
                p=manager.profile
                return self._json(200,dict(api_version=1, flight_enabled=p.allow_flight,
                    vehicle_names=p.names, leader=p.service.leader, shapes=['line','column','vee'],
                    spacing_m={'min':p.safety_distance+.7,'max':8.}, safety_distance_m=p.safety_distance,
                    cruise_z_ned_m=p.settings['Vehicles'][p.service.leader].get('Z',0.)+next(
                        d['target_z'] for d in p.swarm['drones'] if d['vehicle_name']==p.service.leader),
                    supports={'initial_formation':True,'runtime_formation':'requires_B_heartbeat',
                              'pause_resume':False,'spawn_edit':False,'inflight_goal_edit':False,
                              'fleet_edit':False,'speed_altitude_edit':False},
                    flight_limits=p.flight_limits, state_service_url=p.service.url))
            if self.command=='GET' and path=='/api/v1/maps/current':
                return self._json(200,manager.profile.map.public())
            if self.command=='GET' and path=='/api/v1/events':
                after=int(parse_qs(target.query).get('after',['0'])[0])
                return self._events(max(0,after))
            if self.command=='POST' and path=='/api/v1/recovery/check':
                return self._json(200,manager.verify_recovery(self._body()))
            if self.command=='POST' and path=='/api/v1/missions/validate':
                return self._json(200,manager.validate(self._body()))
            if self.command=='POST' and path=='/api/v1/missions':
                return self._json(201,manager.create(self._body()))
            match=re.fullmatch(r'/api/v1/operations/([A-Za-z0-9_-]{1,80})',path)
            if match and self.command=='GET':
                return self._json(200,manager.operation(match[1]))
            match=re.fullmatch(r'/api/v1/missions/([A-Za-z0-9_-]{1,80})(?:/(start|stop|formation))?',path)
            if match:
                mid,action=match.groups()
                if action is None and self.command=='GET':
                    return self._json(200,manager.get(mid))
                if action in {'start','stop'} and self.command=='POST':
                    return self._json(202,getattr(manager,action)(mid,self._body()))
                if action=='formation' and self.command=='PUT':
                    return self._json(202,manager.configure(mid,self._body()))
            raise OperatorError('route_not_found','Unknown API route/method',404)
        except OperatorError as exc:
            self._json(exc.status,dict(error=exc.code,message=str(exc)))
        except (ValueError,TypeError,KeyError):
            self._json(400,dict(error='invalid_request',message='Invalid request fields'))
        except (OSError,TimeoutError):
            return
        except Exception:
            self._json(500,dict(error='internal_error',message='Operator service failed; inspect local logs'))

    def _internal(self,path):
        match=re.fullmatch(r'/api/v1/internal/missions/([A-Za-z0-9_-]{1,80})/formation/(heartbeat|request|result|status)',path)
        require(match is not None,'route_not_found','Unknown internal route',404)
        mid,action=match.groups()
        m=self.server.manager
        if action=='request' and self.command=='GET':
            return self._json(200,m.b_request(mid,self._token()))
        if action=='heartbeat' and self.command=='POST':
            return self._json(200,m.b_heartbeat(mid,self._token(),self._body()))
        if action=='result' and self.command=='POST':
            return self._json(200,m.b_result(mid,self._token(),self._body()))
        if action=='status' and self.command=='POST':
            return self._json(200,m.b_status(mid,self._token(),self._body()))
        raise OperatorError('route_not_found','Unknown method',404)

    def _events(self,after):
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.send_header('Cache-Control','no-store')
        origin=self.headers.get('Origin')
        if origin in self.server.origins:
            self.send_header('Access-Control-Allow-Origin',origin)
        self.end_headers()
        # Bounded connection; reconnect with the cursor, never replay mutations.
        deadline=time.monotonic()+25
        try:
            while time.monotonic()<deadline and not self.server.manager.closed.is_set():
                result=self.server.manager.events_after(after)
                self.wfile.write(('event: update\ndata: '+canonical(result)+'\n\n').encode())
                self.wfile.flush()
                after=result['sequence']
                time.sleep(.5)
        except (OSError,TimeoutError):
            pass
