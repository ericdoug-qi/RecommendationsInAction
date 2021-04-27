# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: as.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import item_info_pb2 as item__info__pb2
import user_info_pb2 as user__info__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='as.proto',
  package='as',
  syntax='proto3',
  serialized_options=b'\200\001\000',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x08\x61s.proto\x12\x02\x61s\x1a\x0fitem_info.proto\x1a\x0fuser_info.proto\"T\n\tASRequest\x12\x0e\n\x06log_id\x18\x01 \x01(\t\x12\x0f\n\x07user_id\x18\x02 \x01(\t\x12&\n\tuser_info\x18\x03 \x01(\x0b\x32\x13.user_info.UserInfo\"\x7f\n\nASResponse\x12#\n\x05\x65rror\x18\x01 \x01(\x0b\x32\x14.as.ASResponse.Error\x12\'\n\nitem_infos\x18\x02 \x03(\x0b\x32\x13.item_info.ItemInfo\x1a#\n\x05\x45rror\x12\x0c\n\x04\x63ode\x18\x01 \x01(\r\x12\x0c\n\x04text\x18\x02 \x01(\t25\n\tASService\x12(\n\x07\x61s_call\x12\r.as.ASRequest\x1a\x0e.as.ASResponseB\x03\x80\x01\x00\x62\x06proto3'
  ,
  dependencies=[item__info__pb2.DESCRIPTOR,user__info__pb2.DESCRIPTOR,])




_ASREQUEST = _descriptor.Descriptor(
  name='ASRequest',
  full_name='as.ASRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='log_id', full_name='as.ASRequest.log_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='user_id', full_name='as.ASRequest.user_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='user_info', full_name='as.ASRequest.user_info', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=50,
  serialized_end=134,
)


_ASRESPONSE_ERROR = _descriptor.Descriptor(
  name='Error',
  full_name='as.ASResponse.Error',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='as.ASResponse.Error.code', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='text', full_name='as.ASResponse.Error.text', index=1,
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
  serialized_start=228,
  serialized_end=263,
)

_ASRESPONSE = _descriptor.Descriptor(
  name='ASResponse',
  full_name='as.ASResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='error', full_name='as.ASResponse.error', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='item_infos', full_name='as.ASResponse.item_infos', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_ASRESPONSE_ERROR, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=263,
)

_ASREQUEST.fields_by_name['user_info'].message_type = user__info__pb2._USERINFO
_ASRESPONSE_ERROR.containing_type = _ASRESPONSE
_ASRESPONSE.fields_by_name['error'].message_type = _ASRESPONSE_ERROR
_ASRESPONSE.fields_by_name['item_infos'].message_type = item__info__pb2._ITEMINFO
DESCRIPTOR.message_types_by_name['ASRequest'] = _ASREQUEST
DESCRIPTOR.message_types_by_name['ASResponse'] = _ASRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ASRequest = _reflection.GeneratedProtocolMessageType('ASRequest', (_message.Message,), {
  'DESCRIPTOR' : _ASREQUEST,
  '__module__' : 'as_pb2'
  # @@protoc_insertion_point(class_scope:as.ASRequest)
  })
_sym_db.RegisterMessage(ASRequest)

ASResponse = _reflection.GeneratedProtocolMessageType('ASResponse', (_message.Message,), {

  'Error' : _reflection.GeneratedProtocolMessageType('Error', (_message.Message,), {
    'DESCRIPTOR' : _ASRESPONSE_ERROR,
    '__module__' : 'as_pb2'
    # @@protoc_insertion_point(class_scope:as.ASResponse.Error)
    })
  ,
  'DESCRIPTOR' : _ASRESPONSE,
  '__module__' : 'as_pb2'
  # @@protoc_insertion_point(class_scope:as.ASResponse)
  })
_sym_db.RegisterMessage(ASResponse)
_sym_db.RegisterMessage(ASResponse.Error)


DESCRIPTOR._options = None

_ASSERVICE = _descriptor.ServiceDescriptor(
  name='ASService',
  full_name='as.ASService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=265,
  serialized_end=318,
  methods=[
  _descriptor.MethodDescriptor(
    name='as_call',
    full_name='as.ASService.as_call',
    index=0,
    containing_service=None,
    input_type=_ASREQUEST,
    output_type=_ASRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_ASSERVICE)

DESCRIPTOR.services_by_name['ASService'] = _ASSERVICE

# @@protoc_insertion_point(module_scope)
