# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rank.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import user_info_pb2 as user__info__pb2
import item_info_pb2 as item__info__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='rank.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\nrank.proto\x1a\x0fuser_info.proto\x1a\x0fitem_info.proto\"n\n\x0bRankRequest\x12\x0e\n\x06log_id\x18\x01 \x01(\t\x12&\n\tuser_info\x18\x02 \x01(\x0b\x32\x13.user_info.UserInfo\x12\'\n\nitem_infos\x18\x03 \x03(\x0b\x32\x13.item_info.ItemInfo\"\xae\x01\n\x0cRankResponse\x12\"\n\x05\x65rror\x18\x01 \x01(\x0b\x32\x13.RankResponse.Error\x12,\n\x0bscore_pairs\x18\x02 \x03(\x0b\x32\x17.RankResponse.ScorePair\x1a#\n\x05\x45rror\x12\x0c\n\x04\x63ode\x18\x01 \x01(\r\x12\x0c\n\x04text\x18\x02 \x01(\t\x1a\'\n\tScorePair\x12\x0b\n\x03nid\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x32:\n\x0bRankService\x12+\n\x0crank_predict\x12\x0c.RankRequest\x1a\r.RankResponseb\x06proto3'
  ,
  dependencies=[user__info__pb2.DESCRIPTOR,item__info__pb2.DESCRIPTOR,])




_RANKREQUEST = _descriptor.Descriptor(
  name='RankRequest',
  full_name='RankRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='log_id', full_name='RankRequest.log_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='user_info', full_name='RankRequest.user_info', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='item_infos', full_name='RankRequest.item_infos', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=48,
  serialized_end=158,
)


_RANKRESPONSE_ERROR = _descriptor.Descriptor(
  name='Error',
  full_name='RankResponse.Error',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='RankResponse.Error.code', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='text', full_name='RankResponse.Error.text', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=259,
  serialized_end=294,
)

_RANKRESPONSE_SCOREPAIR = _descriptor.Descriptor(
  name='ScorePair',
  full_name='RankResponse.ScorePair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='nid', full_name='RankResponse.ScorePair.nid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='score', full_name='RankResponse.ScorePair.score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=296,
  serialized_end=335,
)

_RANKRESPONSE = _descriptor.Descriptor(
  name='RankResponse',
  full_name='RankResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='error', full_name='RankResponse.error', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='score_pairs', full_name='RankResponse.score_pairs', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_RANKRESPONSE_ERROR, _RANKRESPONSE_SCOREPAIR, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=335,
)

_RANKREQUEST.fields_by_name['user_info'].message_type = user__info__pb2._USERINFO
_RANKREQUEST.fields_by_name['item_infos'].message_type = item__info__pb2._ITEMINFO
_RANKRESPONSE_ERROR.containing_type = _RANKRESPONSE
_RANKRESPONSE_SCOREPAIR.containing_type = _RANKRESPONSE
_RANKRESPONSE.fields_by_name['error'].message_type = _RANKRESPONSE_ERROR
_RANKRESPONSE.fields_by_name['score_pairs'].message_type = _RANKRESPONSE_SCOREPAIR
DESCRIPTOR.message_types_by_name['RankRequest'] = _RANKREQUEST
DESCRIPTOR.message_types_by_name['RankResponse'] = _RANKRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RankRequest = _reflection.GeneratedProtocolMessageType('RankRequest', (_message.Message,), {
  'DESCRIPTOR' : _RANKREQUEST,
  '__module__' : 'rank_pb2'
  # @@protoc_insertion_point(class_scope:RankRequest)
  })
_sym_db.RegisterMessage(RankRequest)

RankResponse = _reflection.GeneratedProtocolMessageType('RankResponse', (_message.Message,), {

  'Error' : _reflection.GeneratedProtocolMessageType('Error', (_message.Message,), {
    'DESCRIPTOR' : _RANKRESPONSE_ERROR,
    '__module__' : 'rank_pb2'
    # @@protoc_insertion_point(class_scope:RankResponse.Error)
    })
  ,

  'ScorePair' : _reflection.GeneratedProtocolMessageType('ScorePair', (_message.Message,), {
    'DESCRIPTOR' : _RANKRESPONSE_SCOREPAIR,
    '__module__' : 'rank_pb2'
    # @@protoc_insertion_point(class_scope:RankResponse.ScorePair)
    })
  ,
  'DESCRIPTOR' : _RANKRESPONSE,
  '__module__' : 'rank_pb2'
  # @@protoc_insertion_point(class_scope:RankResponse)
  })
_sym_db.RegisterMessage(RankResponse)
_sym_db.RegisterMessage(RankResponse.Error)
_sym_db.RegisterMessage(RankResponse.ScorePair)



_RANKSERVICE = _descriptor.ServiceDescriptor(
  name='RankService',
  full_name='RankService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=337,
  serialized_end=395,
  methods=[
  _descriptor.MethodDescriptor(
    name='rank_predict',
    full_name='RankService.rank_predict',
    index=0,
    containing_service=None,
    input_type=_RANKREQUEST,
    output_type=_RANKRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_RANKSERVICE)

DESCRIPTOR.services_by_name['RankService'] = _RANKSERVICE

# @@protoc_insertion_point(module_scope)
